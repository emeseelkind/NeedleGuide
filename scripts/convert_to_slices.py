import os
import glob
import argparse
import logging
import numpy as np
import sys
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--use-file-prefix", action="store_true")
    parser.add_argument("--include-unlabeled-frames", action="store_true")
    parser.add_argument(
        "--label-output",
        type=str,
        choices=["preserve-input", "labelmap-only", "both"],
        default="preserve-input",
        help="How to save labels. Default preserves input format.",
    )
    parser.add_argument(
        "--overlap-policy",
        type=str,
        choices=["priority", "argmax"],
        default="priority",
        help="How to resolve overlaps when converting multichannel to labelmap.",
    )
    parser.add_argument("--log_file", type=str)
    return parser.parse_args()


def load_array(path):
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[arr.files[0]]
    return arr


def normalize_segmentation(segmentation_arr):
    if segmentation_arr.ndim == 2:
        segmentation_arr = segmentation_arr[np.newaxis, :, :]
    if segmentation_arr.ndim == 3:
        return segmentation_arr, "single"
    if segmentation_arr.ndim == 4:
        if segmentation_arr.shape[3] == 1:
            return segmentation_arr, "single"
        return segmentation_arr, "multi"
    logging.error(f"Unexpected segmentation shape {segmentation_arr.shape}")
    sys.exit(1)


def multichannel_to_labelmap(seg_frame, overlap_policy):
    if seg_frame.ndim != 3:
        logging.error(f"Expected multichannel frame, got shape {seg_frame.shape}")
        sys.exit(1)
    num_channels = seg_frame.shape[2]
    dtype = np.uint16 if num_channels > 255 else np.uint8
    if overlap_policy == "priority":
        labelmap = np.zeros(seg_frame.shape[:2], dtype=dtype)
        for channel_index in range(num_channels):
            mask = seg_frame[:, :, channel_index] > 0
            labelmap[(labelmap == 0) & mask] = channel_index + 1
        return labelmap
    if overlap_policy == "argmax":
        any_mask = np.any(seg_frame > 0, axis=-1)
        argmax = np.argmax(seg_frame, axis=-1)
        return np.where(any_mask, argmax + 1, 0).astype(dtype)
    logging.error(f"Unknown overlap policy: {overlap_policy}")
    sys.exit(1)


def main(args):
    # Find all data segmentation files and matching ultrasound files in input directory
    ultrasound_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_ultrasound*.npy")))
    segmentation_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_segmentation*.npy")))
    transform_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_transform*.npy")))
    indices_data_files = sorted(glob.glob(os.path.join(args.data_folder, "*_indices*.npy")))

    # Make sure output folder exists and save a copy of the configuration file

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up logging into file or console

    if args.log_file is not None:
        log_file = os.path.join(args.output_dir, args.log_file)
        logging.basicConfig(filename=log_file, filemode="w", level="INFO")
    else:
        logging.basicConfig(level="INFO")  # Log to console

    # Create subfolders for images, segmentations, and transforms if they don't exist
    image_dir = os.path.join(args.output_dir, "images")
    label_dir = os.path.join(args.output_dir, "labels")
    tfm_dir = os.path.join(args.output_dir, "transforms")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(tfm_dir, exist_ok=True)

    logging.info(f"Saving individual images, segmentations, and transforms to {args.output_dir}...")

    for pt_idx in tqdm(range(len(ultrasound_data_files))):
        # Create new directory for individual images
        if args.use_file_prefix:
            pt_image_dir = os.path.join(image_dir, os.path.basename(ultrasound_data_files[pt_idx]).rsplit("_", 1)[0])
            pt_label_dir = os.path.join(label_dir, os.path.basename(ultrasound_data_files[pt_idx]).rsplit("_", 1)[0])
            pt_tfm_dir = os.path.join(tfm_dir, os.path.basename(ultrasound_data_files[pt_idx]).rsplit("_", 1)[0])
        else:
            pt_image_dir = os.path.join(image_dir, f"{pt_idx:04d}")
            pt_label_dir = os.path.join(label_dir, f"{pt_idx:04d}")
            pt_tfm_dir = os.path.join(tfm_dir, f"{pt_idx:04d}")
        os.makedirs(pt_image_dir, exist_ok=True)
        os.makedirs(pt_label_dir, exist_ok=True)
        os.makedirs(pt_tfm_dir, exist_ok=True)

        # Read images, segmentations, transforms, and indices
        ultrasound_arr = load_array(ultrasound_data_files[pt_idx])
        segmentation_arr = load_array(segmentation_data_files[pt_idx])
        segmentation_arr, seg_format = normalize_segmentation(segmentation_arr)
        if seg_format == "multi":
            overlap_pixels = np.sum(np.sum(segmentation_arr > 0, axis=-1) > 1)
            if overlap_pixels > 0:
                logging.info(f"Found {overlap_pixels} overlapping pixels in segmentation channels")
        logging.info(
            f"Segmentation format: {seg_format} (shape {segmentation_arr.shape}), "
            f"label_output={args.label_output}"
        )
        if transform_data_files:
            transform_arr = load_array(transform_data_files[pt_idx])
        if indices_data_files and indices_data_files[pt_idx]:
            indices_arr = load_array(indices_data_files[pt_idx])
        else:
            indices_arr = None

        seg_idx = 0
        for frame_idx in range(ultrasound_arr.shape[0]):
            # Save individual images
            image_fn = os.path.join(pt_image_dir, f"{frame_idx:04d}_ultrasound.npy")
            np.save(image_fn, ultrasound_arr[frame_idx])

            if indices_arr is not None and args.include_unlabeled_frames:
                if frame_idx in indices_arr:
                    # Save individual segmentations
                    seg_frame = segmentation_arr[seg_idx]
                    seg_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_segmentation.npy")
                    if args.label_output == "preserve-input":
                        np.save(seg_fn, seg_frame)
                    elif args.label_output == "labelmap-only":
                        if seg_format == "multi":
                            seg_frame = multichannel_to_labelmap(seg_frame, args.overlap_policy)
                        else:
                            seg_frame = np.squeeze(seg_frame)
                        np.save(seg_fn, seg_frame)
                    else:
                        np.save(seg_fn, seg_frame)
                        if seg_format == "multi":
                            labelmap_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_labelmap.npy")
                            np.save(labelmap_fn, multichannel_to_labelmap(seg_frame, args.overlap_policy))
                        else:
                            labelmap_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_labelmap.npy")
                            np.save(labelmap_fn, np.squeeze(seg_frame))
                    seg_idx += 1
            else:
                seg_frame = segmentation_arr[frame_idx]
                seg_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_segmentation.npy")
                if args.label_output == "preserve-input":
                    np.save(seg_fn, seg_frame)
                elif args.label_output == "labelmap-only":
                    if seg_format == "multi":
                        seg_frame = multichannel_to_labelmap(seg_frame, args.overlap_policy)
                    else:
                        seg_frame = np.squeeze(seg_frame)
                    np.save(seg_fn, seg_frame)
                else:
                    np.save(seg_fn, seg_frame)
                    if seg_format == "multi":
                        labelmap_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_labelmap.npy")
                        np.save(labelmap_fn, multichannel_to_labelmap(seg_frame, args.overlap_policy))
                    else:
                        labelmap_fn = os.path.join(pt_label_dir, f"{frame_idx:04d}_labelmap.npy")
                        np.save(labelmap_fn, np.squeeze(seg_frame))

            # Save individual transforms
            if transform_data_files:
                tfm_fn = os.path.join(pt_tfm_dir, f"{frame_idx:04d}_transform.npy")
                np.save(tfm_fn, transform_arr[frame_idx])
    logging.info(f"Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
