import os
import numpy as np
from tqdm import tqdm
import sys
import nibabel as nib

def center_crop_2d(image, target_size=512):
    """
    Perform center crop on a 2D image to target_size x target_size.
    Handles images that are larger (crop), smaller (pad), or mixed dimensions.
    
    Args:
        image: numpy array (H, W) or (H, W, C)
        target_size: desired output size (default 512)
    
    Returns:
        cropped/padded image of shape (target_size, target_size) or (target_size, target_size, C)
    """
    is_2d = len(image.shape) == 2
    h, w = image.shape[:2]
    
    # Process height: crop if too large, pad if too small
    if h > target_size:
        start_h = (h - target_size) // 2
        image = image[start_h:start_h + target_size, ...]
    elif h < target_size:
        pad_before = (target_size - h) // 2
        pad_after = target_size - h - pad_before
        if is_2d:
            image = np.pad(image, ((pad_before, pad_after), (0, 0)), mode='constant', constant_values=0)
        else:
            image = np.pad(image, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    # Process width: crop if too large, pad if too small
    if w > target_size:
        start_w = (w - target_size) // 2
        image = image[:, start_w:start_w + target_size, ...]
    elif w < target_size:
        pad_before = (target_size - w) // 2
        pad_after = target_size - w - pad_before
        if is_2d:
            image = np.pad(image, ((0, 0), (pad_before, pad_after)), mode='constant', constant_values=0)
        else:
            image = np.pad(image, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant', constant_values=0)
    
    return image


def process_failed_cases(failed_cases=list, source_images_base=None, source_labels_base=None, output_images_dir=None, output_labels_dir=None):
    """
    Process cases that failed during main processing (mixed dimensions like 480x640).
    These are cases that errored out. Use the fixed center_crop_2d function.
    
    Args:
        failed_cases (list): A list of case IDs that failed during the main processing step.
        source_images_base (str): The base directory containing the source image folders.
        source_labels_base (str): The base directory containing the source label folders.
        output_images_dir (str): The directory where processed images will be saved.
        output_labels_dir (str): The directory where processed labels will be saved.
    
    Returns:
        Saves processed .npy files in the nnUNet imagesTr and labelsTr directories with correct naming.
    """
    
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"Processing {len(failed_cases)} failed cases...")
    print(f"Source images: {source_images_base}")
    print(f"Source labels: {source_labels_base}")
    print(f"Output images: {output_images_dir}")
    print(f"Output labels: {output_labels_dir}")
    
    successful = 0
    failed = 0
    
    for case_id in tqdm(failed_cases, desc="Processing failed cases"):
        try:
            # Construct file paths
            image_file = os.path.join(source_images_base, case_id, f"{case_id}_ultrasound.npy")
            label_file = os.path.join(source_labels_base, case_id, f"{case_id}_segmentation.npy")
            
            # Check if both files exist
            if not os.path.exists(image_file):
                print(f"Warning: Image file not found: {image_file}")
                failed += 1
                continue
            
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found: {label_file}")
                failed += 1
                continue
            
            # Load npy files
            image = np.load(image_file)
            label = np.load(label_file)
            
            # Center crop to 512x512 (now with fixed function)
            image_cropped = center_crop_2d(image, target_size=512)
            label_cropped = center_crop_2d(label, target_size=512)
            
            # Verify correct shape
            assert image_cropped.shape == (512, 512), f"Image shape {image_cropped.shape} is not (512, 512)"
            assert label_cropped.shape == (512, 512), f"Label shape {label_cropped.shape} is not (512, 512)"
            
            # Save to output directories with nnUNet naming convention
            output_image_file = os.path.join(output_images_dir, f"Kidney_{case_id}_0000.npy")
            output_label_file = os.path.join(output_labels_dir, f"Kidney_{case_id}.npy")
            
            np.save(output_image_file, image_cropped)
            np.save(output_label_file, label_cropped)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Failed cases processing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"{'='*50}")


def transfer_images(source_images_base=None, source_labels_base=None, output_images_dir=None, output_labels_dir=None):
    """
    Process previously errored images that were manually resized to 512x512.
    Simply transfer and rename them without cropping.

    Args:
        source_images_base (str): The base directory containing the source image folders.
        source_labels_base (str): The base directory containing the source label folders.
        output_images_dir (str): The directory where processed images will be saved.
        output_labels_dir (str): The directory where processed labels will be saved.

    Returns:
        Saves .npy files in the nnUNet imagesTr and labelsTr directories with correct naming.
    """
    
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"Processing errored images...")
    print(f"Source images: {source_images_base}")
    print(f"Source labels: {source_labels_base}")
    print(f"Output images: {output_images_dir}")
    print(f"Output labels: {output_labels_dir}")
    
    # Get all subdirectories (0000, 0001, etc.)
    image_dirs = sorted([d for d in os.listdir(source_images_base) 
                        if os.path.isdir(os.path.join(source_images_base, d))])
    
    if not image_dirs:
        print("No image directories found in errored images folder!")
        return
    
    print(f"\nFound {len(image_dirs)} case directories to transfer")
    
    successful = 0
    failed = 0
    
    for case_id in tqdm(image_dirs, desc="Transferring errored images"):
        try:
            # Construct file paths
            image_file = os.path.join(source_images_base, case_id, f"{case_id}_ultrasound.npy")
            label_file = os.path.join(source_labels_base, case_id, f"{case_id}_segmentation.npy")
            
            # Check if both files exist
            if not os.path.exists(image_file):
                print(f"Warning: Image file not found: {image_file}")
                failed += 1
                continue
            
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found: {label_file}")
                failed += 1
                continue
            
            # Load npy files
            image = np.load(image_file)
            label = np.load(label_file)
            
            # Verify dimensions are 512x512
            assert image.shape == (512, 512), f"Image shape {image.shape} is not (512, 512)"
            assert label.shape == (512, 512), f"Label shape {label.shape} is not (512, 512)"
            
            # Save to output directories with nnUNet naming convention
            output_image_file = os.path.join(output_images_dir, f"Kidney_{case_id}_0000.npy")
            output_label_file = os.path.join(output_labels_dir, f"Kidney_{case_id}.npy")
            
            np.save(output_image_file, image)
            np.save(output_label_file, label)
            
            successful += 1
            
        except Exception as e:
            print(f"Error transferring case {case_id}: {str(e)}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Errored images transfer complete!")
    print(f"Successfully transferred: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"{'='*50}")


def process_manual_v2(start_index=535, source_images_base=None, source_labels_base=None, output_images_dir=None, output_labels_dir=None):
    """
    Import manual dataset versions from manual_kidney_segmentation_V2 (0000..0005 folders).
    Save into nnUNet imagesTr and labelsTr with names Kidney_0535_0000, Kidney_0535, ...

    Args:
        start_index (int): The starting index for naming the processed cases.
        source_images_base (str): The base directory containing the source image folders.
        source_labels_base (str): The base directory containing the source label folders.
        output_images_dir (str): The directory where processed images will be saved.
        output_labels_dir (str): The directory where processed labels will be saved.

    Returns:
        Saves processed .npy files in the nnUNet imagesTr and labelsTr directories with correct naming.
    """
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print(f"Processing manual_v2 images from {source_images_base} and labels from {source_labels_base}")
    print(f"Writing to {output_images_dir} and {output_labels_dir}")

    all_folders = sorted([d for d in os.listdir(source_images_base) if os.path.isdir(os.path.join(source_images_base, d))])
    counter = start_index
    successful = 0
    failed = 0

    for folder in all_folders:
        folder_images_dir = os.path.join(source_images_base, folder)
        folder_labels_dir = os.path.join(source_labels_base, folder)

        if not os.path.isdir(folder_labels_dir):
            print(f"Warning: label folder missing for {folder}")
            continue

        image_files = sorted([f for f in os.listdir(folder_images_dir) if f.endswith("_ultrasound.npy")])

        for image_file in image_files:
            base_name = image_file.replace("_ultrasound.npy", "")
            label_file = f"{base_name}_segmentation.npy"
            image_path = os.path.join(folder_images_dir, image_file)
            label_path = os.path.join(folder_labels_dir, label_file)

            if not os.path.exists(label_path):
                print(f"Warning: label file not found: {label_path}")
                failed += 1
                continue

            try:
                img = np.load(image_path)
                lbl = np.load(label_path)

                img = center_crop_2d(img, target_size=512)
                lbl = center_crop_2d(lbl, target_size=512)

                # If image/label has singleton channel (512,512,1), squeeze to (512,512)
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img[:, :, 0]
                if lbl.ndim == 3 and lbl.shape[2] == 1:
                    lbl = lbl[:, :, 0]

                assert img.shape == (512, 512), f"Image shape {img.shape} is not (512,512)"
                assert lbl.shape == (512, 512), f"Label shape {lbl.shape} is not (512,512)"

                case_id = f"{counter:04d}"
                out_image_file = os.path.join(output_images_dir, f"Kidney_{case_id}_0000.npy")
                out_label_file = os.path.join(output_labels_dir, f"Kidney_{case_id}.npy")

                np.save(out_image_file, img)
                np.save(out_label_file, lbl)

                successful += 1
                counter += 1

            except Exception as e:
                print(f"Error processing manual_v2 case {folder}/{base_name}: {e}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"Manual_v2 processing complete!")
    print(f"Start index: {start_index}")
    print(f"Finished index: {counter-1:04d}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total touched: {successful + failed}")
    print(f"{'='*50}")

def extract_middle_slices_from_3d_volume(raw_img_dir=None, raw_mask_dir=None, out_img_2d=None, out_mask_2d=None):
    """
    Extracts the middle axial slice from 3D NIfTI images and their corresponding masks, and saves them as .npy files for nnU-Net training.
    
    Args:
        raw_img_dir (str): Directory containing the original 3D NIfTI image files.
        raw_mask_dir (str): Directory containing the original 3D NIfTI mask files.
        out_img_2d (str): Directory where the extracted 2D image slices will be saved as .npy files.
        out_mask_2d (str): Directory where the extracted 2D mask slices will be saved as .npy files.

    Returns:
            None
    """
    os.makedirs(out_img_2d, exist_ok=True)
    os.makedirs(out_mask_2d, exist_ok=True)
    
    img_files = [f for f in os.listdir(raw_img_dir) if f.endswith("_imgUS.nii.gz") or f.endswith("_imgUS.nii")]
    
    if not img_files:
        print(f"No files matching '*_imgUS.nii' found in {raw_img_dir}")
        return

    processed = 0
    missing_masks = []

    for img_filename in img_files:
        # Extract the prefix (e.g., '200R' from '200R_imgUS.nii')
        prefix = img_filename.split('_imgUS')[0]
        
        # Construct the expected mask filename
        # Checking for both .nii and .nii.gz just in case
        mask_filename = f"{prefix}_maskUS.nii.gz"
        if not os.path.exists(os.path.join(raw_mask_dir, mask_filename)):
            mask_filename = f"{prefix}_maskUS.nii"

        img_p = os.path.join(raw_img_dir, img_filename)
        mask_p = os.path.join(raw_mask_dir, mask_filename)
        
        if os.path.exists(mask_p):
            try:
                # Load NIfTIs
                img_nifti = nib.load(img_p)
                mask_nifti = nib.load(mask_p)
                
                img_data = img_nifti.get_fdata()
                # Convert 255/nonzero to 1 immediately for nnU-Net compatibility
                mask_data = (mask_nifti.get_fdata() > 0).astype(np.uint8)
                
                if img_data.shape != mask_data.shape:
                    print(f"Skipping {prefix}: Shape mismatch {img_data.shape} vs {mask_data.shape}")
                    continue

                dim_x, dim_y, dim_z = img_data.shape

                # Identify Middle Indices
                mid_x = dim_x // 2  # Sagittal Plane
                mid_z = dim_z // 2  # Axial Plane 
                
                # Save Sagittal Slices
                # np.save(os.path.join(out_img_2d, f"{prefix}_sagittal_img.npy"), img_data[mid_x, :, :])
                # np.save(os.path.join(out_mask_2d, f"{prefix}_sagittal_mask.npy"), mask_data[mid_x, :, :])
                print(f"Saving Axial Shape: {img_data[:, :, mid_z].copy().shape}")
                # Save Axial Slices
                np.save(os.path.join(out_img_2d, f"{prefix}_axial_img.npy"), img_data[:, :, mid_z].copy())
                np.save(os.path.join(out_mask_2d, f"{prefix}_axial_mask.npy"), mask_data[:, :, mid_z].copy())

                print(f"Successfully processed Patient: {prefix}")
                processed += 1
                
            except Exception as e:
                print(f"Error processing {prefix}: {e}")
        else:
            print(f"Warning: Missing mask for {img_filename} (Expected {mask_filename})")
            missing_masks.append(prefix)

    print(f"\n{'='*50}")
    print(f"Middle-slice extraction complete")
    print(f"Processed: {processed}")
    print(f"Missing masks: {len(missing_masks)}")
    if missing_masks:
        print(f"Missing IDs: {', '.join(sorted(missing_masks))}")
    print(f"{'='*50}")

def process_trusted_2d_to_nnunet(start_index=851, source_images_base=None, source_labels_base=None, output_images_dir=None, output_labels_dir=None):
    """
    Process the extracted 2D slices from the TRUSTED dataset and save them in nnU-Net format.
    
    Args:
        start_index (int): The starting index for naming the processed cases.
        source_images_base (str): The base directory containing the 2D image .npy files.
        source_labels_base (str): The base directory containing the 2D mask .npy files.
        output_images_dir (str): The directory where processed images will be saved in nnU-Net format.
        output_labels_dir (str): The directory where processed labels will be saved in nnU-Net format.
        
    Returns:
        Saves processed .npy files in the nnU-Net imagesTr and labelsTr directories with correct naming.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(source_images_base) if f.endswith("_img.npy")])

    counter = start_index
    successful = 0
    failed = 0

    print(f"Processing {len(image_files)} slices from 2D folders...")

    for image_file in image_files:
        # Correct naming logic: 200R_axial_img.npy -> 200R_axial_mask.npy
        base_name = image_file.replace("_img.npy", "")
        label_file = f"{base_name}_mask.npy"
        
        image_path = os.path.join(source_images_base, image_file)
        label_path = os.path.join(source_labels_base, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: Mask not found for {image_file}")
            failed += 1
            continue

        try:
            img = np.load(image_path)
            lbl = np.load(label_path)

            # Standardizing shapes for nnU-Net (512, 512)
            # Assuming you have a center_crop_2d function defined elsewhere
            # If they are already 512,512, this is just a safety check
            if img.shape != (512, 512):
              
                img = center_crop_2d(img, target_size=512)
                lbl = center_crop_2d(lbl, target_size=512)

            # Squeeze channel if it exists (H, W, 1) -> (H, W)
            if img.ndim == 3: img = img.squeeze()
            if lbl.ndim == 3: lbl = lbl.squeeze()

            # Assign unique ID and save
            # Every slice (axial and sagittal) gets its own Kidney_XXXX ID
            case_id = f"{counter:04d}"
            out_img_name = f"Kidney_{case_id}_0000.npy"
            out_lbl_name = f"Kidney_{case_id}.npy"

            np.save(os.path.join(output_images_dir, out_img_name), img)
            np.save(os.path.join(output_labels_dir, out_lbl_name), lbl)

            successful += 1
            counter += 1

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            failed += 1

    print(f"\nProcessing Complete!")
    print(f"Saved {successful} cases (IDs {start_index} to {counter-1})")

def convert_multiclass_to_oneclass(source_labels=None, output_dir=None):
    """
    this script converts 3-class kidney segmentation labels into 1-class labels. The original labels are:
    0 - background
    1 - kidney
    2 - calyx
    3 - fluid
    The converted labels will be:
    0 - background
    1 - kidney (including calyx and fluid)
    Outputs modified .npy files in a specified directory. saved as .npy files. 
    for example oneclass\labels\0000_segmentation.npy
    
    Args:
        source_labels (str): The directory containing the original multiclass label .npy files.
        output_dir (str): The directory where the converted one-class label .npy files will be saved.
    
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing labels from: {source_labels}")
    print(f"Saving modified labels to: {output_dir}")

    # Get all label folders (0000, 0001, etc.)
    label_files = [f for f in os.listdir(source_labels) if f.endswith(".npy")]
    
    print(f"Found {len(label_files)} labels in multiclass folder.")

    processed_count = 0

    for filename in label_files:
        try:
            # Load the original multiclass data
            file_path = os.path.join(source_labels, filename)
            data = np.load(file_path)

            # --- THE CONVERSION ---
            # Anything that isn't background (0) becomes the Kidney class (1).
            # This automatically groups Kidney, Calyx, and Fluid into one label.
            one_class_data = (data > 0).astype(np.uint8)

            # Save to the official labelsTr folder
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, one_class_data)

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nFinished! {processed_count} files saved to {output_dir}")
    print("All non-zero pixels (1, 2, 3) are now Label 1.")

def process_dataset():
    """
    Process kidney ultrasound dataset for nnUNet.
    """
    # Define paths
    # source_images_base = r"P:\data\USKidneySegmentation\kidney_dataset\backup_original_size\images"
    # source_labels_base = r"P:\data\USKidneySegmentation\kidney_dataset\backup_original_size\labels"
    # output_images_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\imagesTr"
    # output_labels_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\labelsTr"
    source_images_base = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_images_test_set256"
    source_labels_base = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_labels_test_set256"
    output_images_dir = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_images_test_set512"
    output_labels_dir = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_labels_test_set512"
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"Source images: {source_images_base}")
    print(f"Source labels: {source_labels_base}")
    print(f"Output images: {output_images_dir}")
    print(f"Output labels: {output_labels_dir}")
    
    if not os.path.exists(source_images_base):
        print(f"Error: The path {source_images_base} does not exist.")
        return

    # 1. First, check if there are subdirectories (0000, 0001, etc.)
    image_dirs = sorted([d for d in os.listdir(source_images_base) 
                        if os.path.isdir(os.path.join(source_images_base, d))])

    if image_dirs:
        print(f"Found {len(image_dirs)} case directories.")
        # Proceed with directory-based logic
        case_ids = image_dirs 
    else:
        # 2. If no directories, check for flat files (0000_ultrasound.npy, etc.)
        image_files = sorted([f for f in os.listdir(source_images_base) 
                             if f.endswith('_ultrasound.npy') and os.path.isfile(os.path.join(source_images_base, f))])
        
        if image_files:
            case_ids = [f.replace('_ultrasound.npy', '') for f in image_files]
            print(f"Found {len(case_ids)} case files.")
        else:
            print("No valid image directories or _ultrasound.npy files found!")
            return
            
    print(f"\nFound {len(image_dirs)} case directories")
    
    successful = 0
    failed = 0
    
    for case_id in tqdm(image_dirs, desc="Processing cases"):
        try:
            # Construct file paths
            image_file = os.path.join(source_images_base, case_id, f"{case_id}_ultrasound.npy")
            label_file = os.path.join(source_labels_base, case_id, f"{case_id}_segmentation.npy")
            
            # Check if both files exist
            if not os.path.exists(image_file):
                print(f"Warning: Image file not found: {image_file}")
                failed += 1
                continue
            
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found: {label_file}")
                failed += 1
                continue
            
            # Load npy files
            image = np.load(image_file)
            label = np.load(label_file)
            
            # Center crop to 512x512
            image_cropped = center_crop_2d(image, target_size=512)
            label_cropped = center_crop_2d(label, target_size=512)
            
            # Ensure correct shape
            assert image_cropped.shape == (512, 512), f"Image shape {image_cropped.shape} is not (512, 512)"
            assert label_cropped.shape == (512, 512), f"Label shape {label_cropped.shape} is not (512, 512)"
            
            # Save to output directories with nnUNet naming convention
            output_image_file = os.path.join(output_images_dir, f"Kidney_{case_id}_0000.npy")
            output_label_file = os.path.join(output_labels_dir, f"Kidney_{case_id}.npy")
            
            np.save(output_image_file, image_cropped)
            np.save(output_label_file, label_cropped)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"{'='*50}")


def convert_npy_to_nifti(target_dir=None, output_dir=None):
    """
    Convert .npy label files to .nii.gz format for nnU-Net compatibility.   
    This is specifically for the manual_v2 dataset where we have .npy files that need to be converted to NIfTI format.
    The function assumes that the .npy files are 2D (H, W)
    and will save them as 3D NIfTI with a singleton dimension (H, W, 1) which is standard for nnU-Net.
    The affine is set to identity since these are 2D slices without real-world spatial information.
    
    Args:
        target_dir (str): The directory containing the .npy files to convert.
        output_dir (str): The directory where the converted .nii.gz files will be saved.
    
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(target_dir) if f.endswith(".npy")]
    
    if not npy_files:
        print(f"No .npy files found in {target_dir}")
        return

    print(f"Found {len(npy_files)} files. Starting conversion to .nii.gz...")

    # Define a standard Identity Affine for 2D slices
    # This is standard for 2D nnU-Net inputs
    affine = np.eye(4)

    processed_count = 0
    for filename in npy_files:
        try:
            # Load the numpy array
            npy_path = os.path.join(target_dir, filename)
            data = np.load(npy_path)

            # Ensure data is the correct type (uint8 for labels)
            data = data.astype(np.uint8)

            # nnU-Net expects 3D NIfTIs even for 2D data
            # We add a singleton dimension to make it (H, W, 1)
            if data.ndim == 2:
                data = data[:, :, np.newaxis]

            # Create the NIfTI object
            nifti_img = nib.Nifti1Image(data, affine)

            # Define output name: Kidney_0000.npy -> Kidney_0000.nii.gz
            nii_filename = filename.replace(".npy", ".nii.gz")
            nii_path = os.path.join(output_dir, nii_filename)

            # Save the NIfTI
            nib.save(nifti_img, nii_path)

            # Optional: Remove the original .npy to save space and avoid confusion
            # os.remove(npy_path) 

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Converted {processed_count} files...")

        except Exception as e:
            print(f"Error converting {filename}: {e}")

    print(f"\nFinished! {processed_count} NIfTI files created in {target_dir}")

def rename_nifti_files(flag, target_dir=None, output_dir=None):
    """
    Rename .nii.gz files to match nnU-Net conventions.

    Args:
        flag (str): "images" or "labels" to indicate which type of files to rename.
        target_dir (str): The directory containing the .nii.gz files to rename.
        output_dir (str): The directory where the renamed .nii.gz files will be saved.

    Returns:
        None
    """
    if flag == "images":
        os.makedirs(output_dir, exist_ok=True)
        nii_files = [f for f in os.listdir(target_dir) if f.endswith(".nii.gz") or f.endswith(".nii")]
        for filename in nii_files:
            try:
                base_name = filename.replace("_ultrasound.nii.gz", "").replace("_ultrasound.nii", "")
                new_name = f"Kidney_{base_name}_0000.nii.gz"
                os.rename(os.path.join(target_dir, filename), os.path.join(output_dir, new_name))
                print(f"Renamed {filename} to {new_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
    elif flag == "labels":
        os.makedirs(output_dir, exist_ok=True)
        nii_files = [f for f in os.listdir(target_dir) if f.endswith(".nii.gz") or f.endswith(".nii")]
        for filename in nii_files:
            try:
                base_name = filename.replace("_segmentation.nii.gz", "").replace("_segmentation.nii", "")
                new_name = f"Kidney_{base_name}.nii.gz"
                os.rename(os.path.join(target_dir, filename), os.path.join(output_dir, new_name))
                print(f"Renamed {filename} to {new_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")


def main():    
    if len(sys.argv) > 1 and sys.argv[1] == "--dataset":
        process_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "--failed":
        process_failed_cases(failed_cases=[], source_images_base=r"P:\data\USKidneySegmentation\kidney_dataset\backup_original_size\images", 
                            source_labels_base=r"P:\data\USKidneySegmentation\kidney_dataset\backup_original_size\labels", 
                            output_images_dir=r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\imagesTr", 
                            output_labels_dir=r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\labelsTr")
    elif len(sys.argv) > 1 and sys.argv[1] == "--errored":
        transfer_images(source_images_base = r"P:\data\USKidneySegmentation\kidney_dataset\images",
                            source_labels_base = r"P:\data\USKidneySegmentation\kidney_dataset\labels",
                            output_images_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\imagesTr",
                            output_labels_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\labelsTr")
    elif len(sys.argv) > 1 and sys.argv[1] == "--manual-v2":
        process_manual_v2(start_index=535, source_images_base = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\images",
                            source_labels_base = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\labels",
                            output_images_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\imagesTr",
                            output_labels_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\labelsTr")
    elif len(sys.argv) > 1 and sys.argv[1] == "--extract-middle-slices-from-3d-volume":
        extract_middle_slices_from_3d_volume(raw_img_dir = r"P:\data\USKidneySegmentation\TRUSTED_Kidney_US\US_images",
                            raw_mask_dir = r"P:\data\USKidneySegmentation\TRUSTED_Kidney_US\US_masks_GT",
                            out_img_2d  = r"P:\data\USKidneySegmentation\TRUSTED_Kidney_US\US_images_2D",
                            out_mask_2d = r"P:\data\USKidneySegmentation\TRUSTED_Kidney_US\US_masks_GT_2D")
    elif len(sys.argv) > 1 and sys.argv[1] == "--trusted-2d-to-nnunet":
        process_trusted_2d_to_nnunet(start_index=851, source_images_base = r"P:\data\USKidneySegmentation\TRUSTED_Kidney_US\US_images_2D",
                            source_labels_base = r"P:\data\USKidneySegmentation\TRUSTED_Kidney_US\US_masks_GT_2D",
                            output_images_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\imagesTr",
                            output_labels_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\labelsTr")
    elif len(sys.argv) > 1 and sys.argv[1] == "--convert-multiclass":
        convert_multiclass_to_oneclass(source_labels = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\labelsTrmulticlass",
                            output_dir = r"P:\data\USKidneySegmentation\nnUNet_raw\Dataset006_KidneyoneclassV2\labelsTr")
    elif len(sys.argv) > 1 and sys.argv[1] == "--convert-npy-to-nifti":
        convert_npy_to_nifti(target_dir = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_labels_test_set256",
                            output_dir = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_labels_test_setnii")
    elif len(sys.argv) > 1 and sys.argv[1] == "--rename-nifti":
        if len(sys.argv) > 2:
            rename_nifti_files(sys.argv[2], target_dir = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_images_test_ds7",
                            output_dir = r"P:\data\USKidneySegmentation\manual_kidney_segmentation\manual_kidney_segmentation_V2\0005_images_test_ds7")
        else:
            print("Please specify the flag for the NIfTI files to rename (images or labels).")

    else:
        print("\nAvailable options:")
        print("\n==============================")
        print("To process the main dataset, run:")
        print("  python prepare_nnunet_dataset006.py --dataset")
        print("\nTo process failed cases with mixed dimensions, run:")
        print("  python prepare_nnunet_dataset006.py --failed --case-ids [] source_images_base output_images_dir source_labels_base output_labels_dir")
        print("\nTo process previously resized images, run:")
        print("  python prepare_nnunet_dataset006.py --transfer source_images_base output_images_dir source_labels_base output_labels_dir")
        print("\nTo import manual_v2 images/labels, run:")
        print("  python prepare_nnunet_dataset006.py --manual-v2 start_index source_images_base output_images_dir source_labels_base output_labels_dir")
        print("\nTo extract middle slices, run:")
        print("  python prepare_nnunet_dataset006.py --extract-middle-slices-from-3d-volume raw_img_dir raw_mask_dir out_img_2d out_mask_2d")
        print("\nTo process trusted 2D slices into nnUNet format, run:")
        print("  python prepare_nnunet_dataset006.py --trusted-2d-to-nnunet start_index source_images_base output_images_dir source_labels_base output_labels_dir")
        print("\nTo convert multiclass labels to oneclass, run:")
        print("  python prepare_nnunet_dataset006.py --convert-multiclass source_labels output_dir")
        print("\nTo convert .npy labels to .nii.gz format, run:")
        print("  python prepare_nnunet_dataset006.py --convert-npy-to-nifti target_dir output_dir")
        print("\nTo rename NIfTI files, run:")
        print("  python prepare_nnunet_dataset006.py --rename-nifti")
        print("\n==============================")

if __name__ == "__main__":
    main()

