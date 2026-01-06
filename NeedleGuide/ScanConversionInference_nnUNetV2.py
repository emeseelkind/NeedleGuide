"""
Implements an OpenIGTLink client that receives ultrasound (pyigtl.ImageMessage) and sends prediction/segmentation (pyigtl.ImageMessage).
Transform messages (pyigtl.TransformMessage) are also received and sent to the server, but the device name is changed by replacing Image to Prediction.
This is done to ensure that the prediction is visualized in the same position as the ultrasound image.

Supports two model formats:
    1. TorchScript models (.pt/.pth) with embedded config.json (legacy)
    2. nnUNet V2 checkpoints (.pth) with separate plans file (--nnunet-plans)

Arguments:
    model: Path to the model file (TorchScript .pt/.pth or nnUNet checkpoint .pth)
    nnunet-plans: Path to nnUNet plans file (nnUNetPlans.json). Required if using nnUNet checkpoint.
    input device name: This is the device name the client is listening to
    output device name: The device name the client outputs to
    host: Server's IP the client connects to.
    input port: Port used for receiving data from the PLUS server over OpenIGTLink
    output port: Port used for sending data to Slicer over OpenIGTLink
"""

import argparse
import cv2
import json
import logging
import numpy as np
import traceback
import sys
import pyigtl
import time
import torch
import yaml

from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay

# nnUNet V2 imports
try:
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    logging.warning("nnUNet V2 not available. Install with: pip install nnunetv2")

ROOT = Path(__file__).parent.resolve()

# CONSTANTS
TRACKING_METHOD_NONE = 0
TRACKING_METHOD_LOCAL = 1
TRACKING_METHOD_GLOBAL = 2


def load_nnunet_config(plans_path):
    """
    Load nnUNet plans file and extract configuration parameters.
    
    Args:
        plans_path (str): Path to nnUNetPlans.json file
        
    Returns:
        dict: Configuration dictionary with keys:
            - shape: [batch, channels, height, width]
            - normalization_mean: mean for Z-score normalization
            - normalization_std: std for Z-score normalization
            - patch_size: [height, width] patch size
            - plans_manager: PlansManager object for model loading
    """
    plans_path = Path(plans_path)
    if not plans_path.is_absolute():
        plans_path = ROOT / plans_path
    
    if not plans_path.exists():
        raise FileNotFoundError(f"nnUNet plans file not found: {plans_path}")
    
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    
    # Extract 2D configuration
    config_2d = plans['configurations']['2d']
    patch_size = config_2d['patch_size']  # [512, 512]
    
    # Extract architecture parameters
    architecture = config_2d['architecture']
    arch_class_name = architecture['network_class_name']
    arch_kwargs = architecture['arch_kwargs']
    arch_kwargs_req_import = architecture.get('_kw_requires_import', [])
    
    # Extract normalization parameters
    norm_schemes = config_2d.get('normalization_schemes', [])
    if 'ZScoreNormalization' in norm_schemes:
        # Get mean and std from foreground intensity properties
        intensity_props = plans.get('foreground_intensity_properties_per_channel', {})
        if '0' in intensity_props:
            norm_mean = float(intensity_props['0']['mean'])
            norm_std = float(intensity_props['0']['std'])
        else:
            logging.warning("No foreground intensity properties found, using default values")
            norm_mean = 0.0
            norm_std = 1.0
    else:
        logging.warning("ZScoreNormalization not found in plans, using default values")
        norm_mean = 0.0
        norm_std = 1.0
    
    # Create config dict compatible with existing code
    config = {
        'shape': [1, 1, patch_size[0], patch_size[1]],  # [batch, channels, height, width]
        'normalization_mean': norm_mean,
        'normalization_std': norm_std,
        'patch_size': patch_size,
        'plans': plans,
        'configuration_name': '2d',
        # Architecture parameters for get_network_from_plans
        'arch_class_name': arch_class_name,
        'arch_kwargs': arch_kwargs,
        'arch_kwargs_req_import': arch_kwargs_req_import
    }
    
    logging.info(f"Loaded nnUNet config: patch_size={patch_size}, norm_mean={norm_mean:.2f}, norm_std={norm_std:.2f}")
    
    return config


# Parse command line arguments
def ScanConversionInference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model file (TorchScript .pt/.pth or nnUNet checkpoint .pth).")
    parser.add_argument("--nnunet-plans", type=str, help="Path to nnUNet plans file (nnUNetPlans.json). Required if using nnUNet checkpoint.")
    parser.add_argument("--global-norm", type=str, help="Path to global normalization file. Optional.")
    parser.add_argument("--scanconversion-config", type=str, help="Path to scan conversion config (.yaml) file. Optional.")
    parser.add_argument("--input-device-name", type=str, default="Image_Image")
    parser.add_argument("--input-tfm-device-name", type=str, default="ImageToReference")
    parser.add_argument("--output-device-name", type=str, default="Prediction")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--input-port", type=int, default=18944)
    parser.add_argument("--output-port", type=int, default=18945)
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file. Optional.")
    try:
        args = parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)

    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
    
    run_client(args)


def run_client(args):
    """
    Runs the client in an infinite loop, waiting for messages from the server. Once a message is received,
    the message is processed and the inference is sent back to the server as a pyigtl ImageMessage.
    """
    input_client = pyigtl.OpenIGTLinkClient(host=args.host, port=args.input_port)
    output_server = pyigtl.OpenIGTLinkServer(port=args.output_port)
    model = None

    # Initialize timer and counters for profiling

    start_time = time.perf_counter()
    preprocess_counter = 0
    preprocess_total_time = 0
    inference_counter = 0
    inference_total_time = 0
    postprocess_counter = 0
    postprocess_total_time = 0
    total_counter = 0
    total_time = 0
    image_message_counter = 0
    transform_message_counter = 0

    # Load pytorch model
    logging.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model if Path(args.model).is_absolute() else ROOT / args.model
    
    # Determine if using nnUNet or TorchScript model
    use_nnunet = args.nnunet_plans is not None
    
    if use_nnunet:
        if not NNUNET_AVAILABLE:
            raise ImportError("nnUNet V2 is required but not installed. Install with: pip install nnunetv2")
        
        # Load nnUNet configuration
        logging.info("Loading nnUNet plans...")
        config = load_nnunet_config(args.nnunet_plans)
        
        # Get model architecture from plans
        logging.info("Creating model architecture from plans...")
        # Determine output channels - typically 2 for binary segmentation (background + foreground)
        # This can be overridden if needed, but 2 is standard for binary segmentation
        output_channels = 2
        
        model = get_network_from_plans(
            arch_class_name=config['arch_class_name'],
            arch_kwargs=config['arch_kwargs'],
            arch_kwargs_req_import=config['arch_kwargs_req_import'],
            input_channels=1,  # Single channel ultrasound
            output_channels=output_channels,
            allow_init=True,
            deep_supervision=False
        )
        
        # Load checkpoint
        logging.info(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'network_weights' in checkpoint:
                model.load_state_dict(checkpoint['network_weights'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the dict itself is the state dict
                model.load_state_dict(checkpoint)
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # nnUNet models don't use tracking
        method = TRACKING_METHOD_NONE
        logging.info("nnUNet model loaded (no tracking support)")
    else:
        # Load TorchScript model (backward compatibility)
        logging.info("Loading TorchScript model...")
        extra_files = {"config.json": ""}
        model = torch.jit.load(str(model_path), _extra_files=extra_files).to(device)
        config = json.loads(extra_files["config.json"])
        
        # check if tracking data is used for input
        try:
            use_tracking = config["use_tracking_layer"]
            if use_tracking:
                if config["tracking_method"] == "local":
                    # TODO: Implement local tracking inference
                    method = TRACKING_METHOD_LOCAL
                    window_size = config["shape"][1]
                    window_target_frame = config["window_target_frame"]
                    image_pixel_norm = config["orig_img_size"]
                elif config["tracking_method"] == "global":
                    method = TRACKING_METHOD_GLOBAL
                else:
                    method = TRACKING_METHOD_NONE
            else:
                method = TRACKING_METHOD_NONE
        except KeyError:
            logging.info("No tracking data used for input")
            method = TRACKING_METHOD_NONE
    
    logging.info("Model loaded")
    torch.inference_mode()

    # If scan conversion is enabled, compute x_cart, y_cart, vertices, and weights for conversion and interpolation
    if args.scanconversion_config:
        logging.info("Loading scan conversion config...")
        with open(args.scanconversion_config, "r") as f:
            scanconversion_config = yaml.safe_load(f)
        x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        logging.info("Scan conversion config loaded")
    else:
        scanconversion_config = None
        x_cart = None
        y_cart = None
        logging.info("Scan conversion config not found")

    if x_cart is not None and y_cart is not None:
        vertices, weights = scan_interpolation_weights(scanconversion_config)
        mask_array = curvilinear_mask(scanconversion_config)
    else:
        vertices = None
        weights = None
        mask_array = None

    while True:
        image_ready = False
        tfm_ready = False

        # Print average inference time every second
        if time.perf_counter() - start_time > 1.0:
            logging.info("--------------------------------------------------")
            logging.info(f"Image messages received:     {image_message_counter}")
            logging.info(f"Transform messages received: {transform_message_counter}")
            if preprocess_counter > 0:
                avg_preprocess_time = round((preprocess_total_time / preprocess_counter) * 1000, 1)
                logging.info(f"Average preprocess time:     {avg_preprocess_time} ms")
            if inference_counter > 0:
                avg_inference_time = round((inference_total_time / inference_counter) * 1000, 1)
                logging.info(f"Average inference time:      {avg_inference_time} ms")
            if postprocess_counter > 0:
                avg_postprocess_time = round((postprocess_total_time / postprocess_counter) * 1000, 1)
                logging.info(f"Average postprocess time:    {avg_postprocess_time} ms")
            if total_counter > 0:
                avg_total_time = round((total_time / total_counter) * 1000, 1)
                logging.info(f"Average total time:          {avg_total_time} ms")
            start_time = time.perf_counter()
            preprocess_counter = 0
            preprocess_total_time = 0
            inference_counter = 0
            inference_total_time = 0
            postprocess_counter = 0
            postprocess_total_time = 0
            total_counter = 0
            total_time = 0
            image_message_counter = 0
            transform_message_counter = 0
        
        # Receive messages from server
        total_start_time = time.perf_counter()
        image_message = input_client.wait_for_message(args.input_device_name)
        image_message_counter += 1

        orig_img_size = image_message.image.shape
        if model is None:
            logging.error("Model not loaded. Exiting...")
            break

        # Preprocess input
        preprocess_start_time = time.perf_counter()
        # Determine input size from config
        if "patch_size" in config:
            input_size = config["patch_size"][0]  # Use patch_size for nnUNet
        else:
            input_size = config["shape"][-1]  # Use shape for TorchScript
        image = preprocess_image(image_message.image, input_size, config, scanconversion_config, x_cart, y_cart).to(device)
        preprocess_total_time += time.perf_counter() - preprocess_start_time
        preprocess_counter += 1
        image_ready = True

        # Receive and preprocess transform
        tfm_message = input_client.wait_for_message(args.input_tfm_device_name)
        output_tfm_name = tfm_message.device_name.replace("Image", "Pred")
        transform = tfm_message.matrix
        transform_pre = preprocess_transform(transform, method, args, config)
        if transform_pre is not None and isinstance(transform_pre, torch.Tensor):
            transform_pre = transform_pre.to(device)
        transform_message_counter += 1
        tfm_ready = True

        # Check if both image and transform are received
        if image_ready and tfm_ready:
            # Run inference
            inference_start_time = time.perf_counter()
            if method != TRACKING_METHOD_NONE:
                prediction = model((image, transform_pre))
            else:
                prediction = model(image)                
            if isinstance(prediction, list):
                prediction = prediction[0]
            inference_total_time += time.perf_counter() - inference_start_time
            inference_counter += 1

            # Postprocess prediction
            postprocess_start_time = time.perf_counter()
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            prediction = postprocess_prediction(prediction, orig_img_size, scanconversion_config, vertices, weights, mask_array)
            # save prediction for debugging
            postprocess_total_time += time.perf_counter() - postprocess_start_time
            postprocess_counter += 1

            image_message = pyigtl.ImageMessage(prediction, device_name=args.output_device_name)
            output_server.send_message(image_message, wait=True)

            tfm_message = pyigtl.TransformMessage(transform, device_name=output_tfm_name)
            output_server.send_message(tfm_message, wait=True)
            
            total_time += time.perf_counter() - total_start_time
            total_counter += 1


def preprocess_image(image, input_size, config, scanconversion_config=None, x_cart=None, y_cart=None):
    """
    Preprocess image for model input.
    
    Args:
        image: Input image array
        input_size: Target size for resizing
        config: Configuration dict (may contain normalization_mean and normalization_std for nnUNet)
        scanconversion_config: Optional scan conversion config
        x_cart, y_cart: Optional cartesian coordinates for scan conversion
        
    Returns:
        Preprocessed image tensor
    """
    if scanconversion_config is not None:
        # Scan convert image from curvilinear to linear
        num_samples = scanconversion_config["num_samples_along_lines"]
        num_lines = scanconversion_config["num_lines"]
        converted_image = np.zeros((1, num_lines, num_samples))
        converted_image[0, :, :] = map_coordinates(image[0, :, :], [x_cart, y_cart], order=1, mode='constant', cval=0.0)
        # Squeeze converted image to remove channel dimension
        converted_image = converted_image.squeeze()
    else:
        converted_image = cv2.resize(image[0, :, :], (input_size, input_size))  # default is bilinear
    
    # Convert to float tensor
    converted_image = torch.from_numpy(converted_image).unsqueeze(0).unsqueeze(0).float()
    
    # Apply normalization
    if "normalization_mean" in config and "normalization_std" in config:
        # Z-score normalization for nnUNet
        norm_mean = config["normalization_mean"]
        norm_std = config["normalization_std"]
        converted_image = (converted_image - norm_mean) / norm_std
    else:
        # Simple scaling for TorchScript models (backward compatibility)
        converted_image = converted_image / 255.0
    
    return converted_image


def preprocess_transform(transform, method, args, config):
    if method == TRACKING_METHOD_LOCAL:
        raise NotImplementedError("Local tracking inference not implemented")
    elif method == TRACKING_METHOD_GLOBAL:
        try:
            image_to_norm = np.load(args.global_norm)
            transform = image_to_norm @ transform
            transform = np.expand_dims(transform, axis=0)  # add image channel dimension
            transform = torch.from_numpy(transform).unsqueeze(0).float()  # add batch dimension
            return transform
        except AttributeError:
            logging.warning("Global normalization file not found, ignoring tracking data input")
            return transform
    else:
        return transform


def postprocess_prediction(prediction, original_size, scanconversion_config=None, vertices=None, weights=None, mask_array=None):
    """
    Postprocess model prediction to output format.
    
    Args:
        prediction: Model output after softmax [batch, num_classes, height, width]
        original_size: Original image size tuple
        scanconversion_config: Optional scan conversion config
        vertices, weights: Optional interpolation weights for scan conversion
        mask_array: Optional mask array
        
    Returns:
        Postprocessed prediction as uint8 array [1, height, width]
    """
    # prediction shape after softmax: [batch, num_classes, height, width]
    # prediction[0] is background, prediction[1] is foreground class
    prediction = prediction.squeeze().detach().cpu().numpy() * 255
    prediction = np.clip(prediction, 0, 255)
    
    if scanconversion_config is not None:
        # Scan convert prediction from linear to curvilinear
        prediction = scan_convert(prediction[1], scanconversion_config, vertices, weights)
        if mask_array is not None:
            prediction = prediction * mask_array
        prediction = prediction.astype(np.uint8)[np.newaxis, ...]
    else:
        # Resize to original image size
        prediction = cv2.resize(prediction[1], (original_size[2], original_size[1]))
        prediction = prediction.astype(np.uint8)[np.newaxis, ...]
    return prediction


def scan_conversion_inverse(scanconversion_config):
    """
    Compute cartesian coordianates for inverse scan conversion.
    Mapping from curvilinear image to a rectancular image of scan lines as columns.
    The returned cartesian coordinates can be used to map the curvilinear image to a rectangular image using scipy.ndimage.map_coordinates.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Rerturns:
        x_cart (np.ndarray): x coordinates of the cartesian grid.
        y_cart (np.ndarray): y coordinates of the cartesian grid.

    Example:
        >>> x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        >>> scan_converted_image = map_coordinates(ultrasound_data[0, :, :, 0], [x_cart, y_cart], order=3, mode="nearest")
        >>> scan_converted_segmentation = map_coordinates(segmentation_data[0, :, :, 0], [x_cart, y_cart], order=0, mode="nearest")
    """

    # Create sampling points in polar coordinates
    initial_radius = np.deg2rad(scanconversion_config["angle_min_degrees"])
    final_radius = np.deg2rad(scanconversion_config["angle_max_degrees"])
    radius_start_px = scanconversion_config["radius_start_pixels"]
    radius_end_px = scanconversion_config["radius_end_pixels"]

    theta, r = np.meshgrid(np.linspace(initial_radius, final_radius, scanconversion_config["num_samples_along_lines"]),
                           np.linspace(radius_start_px, radius_end_px, scanconversion_config["num_lines"]))

    # Convert the polar coordinates to cartesian coordinates
    x_cart = r * np.cos(theta) + scanconversion_config["center_coordinate_pixel"][0]
    y_cart = r * np.sin(theta) + scanconversion_config["center_coordinate_pixel"][1]

    return x_cart, y_cart


def scan_interpolation_weights(scanconversion_config):
    image_size = scanconversion_config["curvilinear_image_size"]

    x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
    triangulation = Delaunay(np.vstack((x_cart.flatten(), y_cart.flatten())).T)

    grid_x, grid_y = np.mgrid[0:image_size, 0:image_size]
    simplices = triangulation.find_simplex(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
    vertices = triangulation.simplices[simplices]

    X = triangulation.transform[simplices, :2]
    Y = np.vstack((grid_x.flatten(), grid_y.flatten())).T - triangulation.transform[simplices, 2]
    b = np.einsum('ijk,ik->ij', X, Y)
    weights = np.c_[b, 1 - b.sum(axis=1)]

    return vertices, weights


def scan_convert(linear_data, scanconversion_config, vertices, weights):
    """
    Scan convert a linear image to a curvilinear image.

    Args:
        linear_data (np.ndarray): Linear image to be scan converted.
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Returns:
        scan_converted_image (np.ndarray): Scan converted image.
    """
    
    z = linear_data.flatten()
    zi = np.einsum('ij,ij->i', np.take(z, vertices), weights)

    image_size = scanconversion_config["curvilinear_image_size"]
    return zi.reshape(image_size, image_size)


def curvilinear_mask(scanconversion_config):
    """
    Generate a binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Returns:
        mask_array (np.ndarray): Binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.
    """
    angle1 = 90.0 + (scanconversion_config["angle_min_degrees"])
    angle2 = 90.0 + (scanconversion_config["angle_max_degrees"])
    center_rows_px = scanconversion_config["center_coordinate_pixel"][0]
    center_cols_px = scanconversion_config["center_coordinate_pixel"][1]
    radius1 = scanconversion_config["radius_start_pixels"]
    radius2 = scanconversion_config["radius_end_pixels"]
    image_size = scanconversion_config["curvilinear_image_size"]

    mask_array = np.zeros((image_size, image_size), dtype=np.int8)
    mask_array = cv2.ellipse(mask_array, (center_cols_px, center_rows_px), (radius2, radius2), 0.0, angle1, angle2, 1, -1)
    mask_array = cv2.circle(mask_array, (center_cols_px, center_rows_px), radius1, 0, -1)
    # Convert mask_array to uint8
    mask_array = mask_array.astype(np.uint8)

    # Repaint the borders of the mask to zero to allow erosion from all sides
    mask_array[0, :] = 0
    mask_array[:, 0] = 0
    mask_array[-1, :] = 0
    mask_array[:, -1] = 0
    
    # Erode mask by 10 percent of the image size to remove artifacts on the edges
    erosion_size = int(0.1 * image_size)
    mask_array = cv2.erode(mask_array, np.ones((erosion_size, erosion_size), np.uint8), iterations=1)
    
    return mask_array


if __name__ == "__main__":
    ScanConversionInference()
