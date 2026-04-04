import SimpleITK as sitk
import numpy as np
import os
import glob
import csv
from itertools import combinations
import argparse


THRESHOLD = 127  # binarization threshold (0–255)


def load(predictions_dir):
    """Load a float32 volume from the given path.
    
    Args:
        path (str): Path to the NRRD or NIfTI file.
    Returns:
        sitk.Image: Loaded image as SimpleITK Image (float32).
    """
    nrrd_files = sorted(
        glob.glob(os.path.join(predictions_dir, "*.nrrd")) +
        glob.glob(os.path.join(predictions_dir, "*.nii.gz"))
    )
    if len(nrrd_files) < 2:
        raise RuntimeError(f"Need at least 2 files in {predictions_dir}, found {len(nrrd_files)}")

    print(f"Found {len(nrrd_files)} prediction volumes:")
    for f in nrrd_files:
        print(f"  {os.path.basename(f)}")
    return nrrd_files


def load_and_resample(path, reference=None):
    """
    Load an image and resample it to match the reference grid if provided.

    Args:
        path (str): Path to the image file (NRRD or NIfTI).
        reference (sitk.Image, optional): Reference image to resample onto. If None the image is loaded without resampling.
    Returns:
        sitk.Image: Loaded and resampled image as SimpleITK Image (float32).
    """
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    if reference is not None:
        img = sitk.Resample(img, reference, sitk.Transform(),
                            sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return img


def binarize(img, threshold=THRESHOLD):
    """
    Binarize the image using the given threshold.

    Args:
        img (sitk.Image): Input image to binarize.
        threshold (float): Threshold value for binarization (default: 127).
    Returns:
        sitk.Image: Binarized image (float32 with values 0.0 or 1.0).
    """
    return sitk.Cast(img > threshold, sitk.sitkFloat32)


def get_centroid_mm(binary_img):
    """
    Compute the centroid of the binary mask in physical space (mm).
    
    Args:
        binary_img (sitk.Image): Binary image where foreground voxels are non-zero.
    Returns:
        tuple: Centroid coordinates (x, y, z) in physical space (mm), or None if no foreground voxels.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    label_img = sitk.Cast(binary_img, sitk.sitkUInt8)
    stats.Execute(label_img)
    if 1 not in stats.GetLabels():
        return None
    return stats.GetCentroid(1)


def compute_ssd(img_a, img_b):
    """
    Compute Sum of Squared Differences (SSD) between two images.

    Args:
        img_a (sitk.Image): First image.
        img_b (sitk.Image): Second image.
    Returns:
        float: Sum of Squared Differences.
    """
    diff = sitk.GetArrayFromImage(img_a).astype(np.float64) \
         - sitk.GetArrayFromImage(img_b).astype(np.float64)
    return float(np.sum(diff ** 2))


def compute_mse(img_a, img_b):
    """
    Compute Mean Squared Error (MSE) between two images.
    
    Args:
        img_a (sitk.Image): First image.
        img_b (sitk.Image): Second image.
    Returns:
        float: Mean Squared Error.
    """
    diff = sitk.GetArrayFromImage(img_a).astype(np.float64) \
         - sitk.GetArrayFromImage(img_b).astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_dice(binary_a, binary_b):
    """Compute Dice similarity coefficient between two binary images.

    Args:
        binary_a (sitk.Image): First binary image (float32 with values 0.0 or 1.0).
        binary_b (sitk.Image): Second binary image (float32 with values 0.0 or 1.0).
    Returns:
        float: Dice similarity coefficient (0.0 to 1.0).
    """
    arr_a = sitk.GetArrayFromImage(binary_a).astype(np.uint8)
    arr_b = sitk.GetArrayFromImage(binary_b).astype(np.uint8)
    intersection = np.sum(arr_a * arr_b)
    sum_ab = np.sum(arr_a) + np.sum(arr_b)
    if sum_ab == 0:
        return 1.0
    return float(2.0 * intersection / sum_ab)


def compute_ncc(img_a, img_b):
    """
    Compute Normalized Cross-Correlation (NCC) between two images.
    
    Args:
        img_a (sitk.Image): First image.
        img_b (sitk.Image): Second image.
    Returns:
        float: Normalized Cross-Correlation.
    """
    a = sitk.GetArrayFromImage(img_a).astype(np.float64).ravel()
    b = sitk.GetArrayFromImage(img_b).astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compute_hausdorff(binary_a, binary_b):
    """
    Compute symmetric and average Hausdorff distance between two binary images.
    
    Args:
        binary_a (sitk.Image): First binary image (float32 with values 0.0 or 1.0).
        binary_b (sitk.Image): Second binary image (float32 with values 0.0 or 1.0).
    Returns:
        tuple: (symmetric Hausdorff distance in mm, average Hausdorff distance in mm)
    """
    hf = sitk.HausdorffDistanceImageFilter()
    hf.Execute(
        sitk.Cast(binary_a, sitk.sitkUInt8),
        sitk.Cast(binary_b, sitk.sitkUInt8)
    )
    return hf.GetHausdorffDistance(), hf.GetAverageHausdorffDistance()


def centroid_displacement_mm(c_a, c_b):
    """
    Compute Euclidean distance between two centroids in mm.

    Args:
        c_a (tuple): Centroid of first image (x, y, z) in mm.
        c_b (tuple): Centroid of second image (x, y, z) in mm.
    Returns:
        float: Euclidean distance in mm, or None if either centroid is None.
    """
    if c_a is None or c_b is None:
        return None
    return float(np.linalg.norm(np.array(c_a) - np.array(c_b)))


def stem(path):
    """
    Get the filename without extension(s), e.g. 'Prediction-final-12L' from 'Prediction-final-12L.nii.gz'.
    
    Args:
        path (str): Path to the file.
    Returns:
        str: Filename without extension(s).
    """
    name = os.path.basename(path)
    return name[:-7] if name.endswith(".nii.gz") else os.path.splitext(name)[0]



def load_volumes(nrrd_files):
    """
    Load all volumes and their binarized versions, and print their properties.
    
    Args:
        nrrd_files (list): List of file paths to load.
    Returns:
        tuple: (volumes, binaries, names)
            volumes: List of loaded SimpleITK images (float32).
            binaries: List of binarized SimpleITK images (float32).
            names: List of filenames without extensions.
    """
    print("\nLoading volumes...")
    reference_img = load_and_resample(nrrd_files[0])
    volumes  = [reference_img] + [load_and_resample(f, reference_img) for f in nrrd_files[1:]]
    binaries = [binarize(v) for v in volumes]
    names    = [stem(f) for f in nrrd_files]
    return volumes, binaries, names


def compute_centroids(names, binaries):
    """
    Compute centroids of all binary masks and their displacements from the mean centroid.
    
    Args:
        names (list): List of volume names.
        binaries (list): List of binary images.
    Returns:
        tuple: (centroids, valid, displacements, mean_c)
            centroids: List of centroid coordinates (x, y, z) in mm or None for empty masks.
            valid: List of (centroid, name) tuples for non-empty masks.
            displacements: List of centroid displacements from mean in mm, or None if not computable.
            mean_c: Mean centroid coordinates (x, y, z) in mm, or None if not computable.
    """
    print("\n=== Per-volume centroid (mm) ===")
    centroids = []
    for name, binary in zip(names, binaries):
        c = get_centroid_mm(binary)
        centroids.append(c)
        if c:
            print(f"  {name}: x={c[0]:.2f}  y={c[1]:.2f}  z={c[2]:.2f} mm")
        else:
            print(f"  {name}: EMPTY MASK — check threshold or volume")

    valid = [(c, n) for c, n in zip(centroids, names) if c is not None]
    displacements = None
    mean_c = None

    if len(valid) >= 2:
        centroid_arr = np.array([c for c, _ in valid])
        mean_c = centroid_arr.mean(axis=0)
        print(f"\n  Mean centroid: x={mean_c[0]:.2f}  y={mean_c[1]:.2f}  z={mean_c[2]:.2f} mm")
        displacements = np.linalg.norm(centroid_arr - mean_c, axis=1)
        print(f"  Displacement from mean per run:")
        for (_, n), d in zip(valid, displacements):
            print(f"    {n}: {d:.2f} mm")
        print(f"  Max displacement: {displacements.max():.2f} mm")
        print(f"  Mean ± std:       {displacements.mean():.2f} ± {displacements.std():.2f} mm")

    return centroids, valid, displacements, mean_c


def compute_pairwise_metrics(names, volumes, binaries, centroids):
    """
    Compute pairwise metrics (SSD, MSE, NCC, Dice, Hausdorff, centroid distance) between all pairs of volumes.
    Args:
        names (list): List of volume names.
        volumes (list): List of loaded SimpleITK images (float32).
        binaries (list): List of binarized SimpleITK images (float32).
        centroids (list): List of centroid coordinates (x, y, z) in mm or None for empty masks.
    Returns:
        list: List of dictionaries containing metrics for each pair.
    """
    print("\n=== Pairwise metrics ===")
    pair_results = []

    for (i, j) in combinations(range(len(volumes)), 2):
        name_a, name_b = names[i], names[j]
        vol_a,  vol_b  = volumes[i], volumes[j]
        bin_a,  bin_b  = binaries[i], binaries[j]

        ssd            = compute_ssd(vol_a, vol_b)
        mse            = compute_mse(vol_a, vol_b)
        ncc            = compute_ncc(vol_a, vol_b)
        dice           = compute_dice(bin_a, bin_b)
        hd_sym, hd_avg = compute_hausdorff(bin_a, bin_b)
        disp           = centroid_displacement_mm(centroids[i], centroids[j])

        print(f"\n  {name_a}  vs  {name_b}")
        print(f"    SSD:                  {ssd:,.1f}")
        print(f"    MSE:                  {mse:.4f}")
        print(f"    NCC:                  {ncc:.6f}")
        print(f"    Dice:                 {dice:.6f}")
        print(f"    Hausdorff (sym):      {hd_sym:.2f} mm")
        print(f"    Hausdorff (avg):      {hd_avg:.2f} mm")
        if disp is not None:
            print(f"    Centroid distance:    {disp:.2f} mm")
        else:
            print(f"    Centroid distance:    N/A (empty mask)")

        pair_results.append({
            "pair":             f"{name_a} vs {name_b}",
            "SSD":              ssd,
            "MSE":              mse,
            "NCC":              ncc,
            "Dice":             dice,
            "HD_symmetric_mm":  hd_sym,
            "HD_average_mm":    hd_avg,
            "centroid_dist_mm": disp if disp is not None else "",
        })

    return pair_results


def print_summary(pair_results):
    print("\n=== Summary across all pairs ===")
    numeric_fields = ["SSD", "MSE", "NCC", "Dice",
                      "HD_symmetric_mm", "HD_average_mm", "centroid_dist_mm"]
    for metric in numeric_fields:
        vals = [r[metric] for r in pair_results if r[metric] != ""]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            print(f"  {metric:25s}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}")


def save_csv(output_dir, pair_results, valid, displacements):
    """
    Save the pairwise metrics and centroid displacements to a CSV file.
    
    Args:
        output_dir (str): Directory to save the CSV file.
        pair_results (list): List of dictionaries containing metrics for each pair.
        valid (list): List of (centroid, name) tuples for non-empty masks.
        displacements (list): List of centroid displacements from mean in mm, or None if not computable.
    Returns:
        None
    """
    numeric_fields = ["SSD", "MSE", "NCC", "Dice",
                      "HD_symmetric_mm", "HD_average_mm", "centroid_dist_mm"]
    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    fieldnames = ["pair"] + numeric_fields

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_results)

        writer.writerow({k: "" for k in fieldnames})
        writer.writerow({"pair": "SUMMARY (mean)"} | {
            m: f"{np.mean([r[m] for r in pair_results if r[m] != '']):.4f}"
            for m in numeric_fields
        })
        writer.writerow({"pair": "SUMMARY (std)"} | {
            m: f"{np.std([r[m] for r in pair_results if r[m] != '']):.4f}"
            for m in numeric_fields
        })

        if displacements is not None and len(valid) >= 2:
            writer.writerow({k: "" for k in fieldnames})
            writer.writerow({"pair": "--- Per-volume centroids ---"})
            writer.writerow({"pair": "Volume", "SSD": "x_mm", "MSE": "y_mm",
                             "NCC": "z_mm", "HD_symmetric_mm": "disp_from_mean_mm"})
            for (c, n), d in zip(valid, displacements):
                writer.writerow({"pair": n, "SSD": f"{c[0]:.3f}", "MSE": f"{c[1]:.3f}",
                                 "NCC": f"{c[2]:.3f}", "HD_symmetric_mm": f"{d:.3f}"})
        else:
            writer.writerow({k: "" for k in fieldnames})
            writer.writerow({"pair": "--- Per-volume centroids: all masks empty, nothing to report ---"})

    print(f"\nResults saved to: {csv_path}")


def run_comparison(predictions_dir, output_dir):
    """
    Run the full comparison pipeline: load volumes, compute centroids, compute pairwise metrics, print summary, and save to CSV.
    
    Args:
        predictions_dir (str): Directory containing the prediction volumes (NRRD or NIfTI).
        output_dir (str): Directory to save the output CSV file.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    nrrd_files                          = load(predictions_dir)
    volumes, binaries, names            = load_volumes(nrrd_files)
    centroids, valid, displacements, _  = compute_centroids(names, binaries)
    pair_results                        = compute_pairwise_metrics(names, volumes, binaries, centroids)

    print_summary(pair_results)
    save_csv(output_dir, pair_results, valid, displacements)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run prediction comparison")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Path to predictions folder")
    parser.add_argument("--output_dir",      type=str, required=True, help="Path to output folder")
    args = parser.parse_args()

    run_comparison(args.predictions_dir, args.output_dir)


if __name__ == "__main__":
    main()