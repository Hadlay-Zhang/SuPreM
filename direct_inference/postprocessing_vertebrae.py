# Post-processing for vertebrae segmentation masks
# Author: Zhilin Zhang

import os
import glob
import argparse
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import torch
from monai.transforms import KeepLargestConnectedComponent, FillHoles

# Derived from ./dataset/dataloader_test.py
selected_class_map = {
    1: "vertebrae_L5", 2: "vertebrae_L4", 3: "vertebrae_L3", 4: "vertebrae_L2", 5: "vertebrae_L1",
    6: "vertebrae_T12", 7: "vertebrae_T11", 8: "vertebrae_T10", 9: "vertebrae_T9", 10: "vertebrae_T8",
    11: "vertebrae_T7", 12: "vertebrae_T6", 13: "vertebrae_T5", 14: "vertebrae_T4", 15: "vertebrae_T3",
    16: "vertebrae_T2", 17: "vertebrae_T1", 18: "vertebrae_C7", 19: "vertebrae_C6", 20: "vertebrae_C5", 
    21: "vertebrae_C4", 22: "vertebrae_C3", 23: "vertebrae_C2", 24: "vertebrae_C1"
}

# MONAI transforms
# foreground label 1, connectivity=3
keep_largest_cc = KeepLargestConnectedComponent(applied_labels=[1], connectivity=3)
fill_holes_transform = FillHoles(applied_labels=[1], connectivity=3) 

def get_component_volume_cpu(mask_data):
    binary_mask = mask_data > 0
    labeled_array, num_features = ndimage.label(binary_mask)
    volume = np.sum(binary_mask)
    return num_features, volume

def postprocess_single_mask_gpu(input_mask_path, output_mask_path, device):
    """
    Post-processes a single NIfTI mask with MONAI transforms via two logics: 
    1) keep the largest connected component;
    2) fill holes.
    """
    try:
        nii_img = nib.load(input_mask_path)
        original_affine = nii_img.affine
        mask_data_np = nii_img.get_fdata().astype(np.uint8)

        # evaluation
        original_components, original_volume = get_component_volume_cpu(mask_data_np)

        # prepare for GPU
        mask_tensor = torch.from_numpy(mask_data_np)
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor_gpu = mask_tensor.to(device)

        # keep largest connected component
        processed_tensor_gpu = keep_largest_cc(mask_tensor_gpu)
        # fill holes
        processed_tensor_gpu = fill_holes_transform(processed_tensor_gpu)

        # post GPU
        processed_tensor_cpu = processed_tensor_gpu.cpu()
        processed_tensor_cpu = processed_tensor_cpu.squeeze(0)
        processed_mask_np = processed_tensor_cpu.numpy().astype(np.uint8)
        del mask_tensor_gpu
        del processed_tensor_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # evaluation metric
        processed_components, processed_volume = get_component_volume_cpu(processed_mask_np)
        # save results
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        processed_nii = nib.Nifti1Image(processed_mask_np, original_affine)
        nib.save(processed_nii, output_mask_path)

        return original_components, processed_components, original_volume, processed_volume

    except FileNotFoundError:
        print(f"Error: Input file not found {input_mask_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error processing file {input_mask_path} on device {device}: {e}")
        if 'cuda' in str(e).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, None, None

def main(input_dir, output_dir):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # input segmentation masks
    case_dirs = [d for d in glob.glob(os.path.join(input_dir, 'BDMAP_*')) if os.path.isdir(d)]
    if not case_dirs:
        print(f"Error: No 'BDMAP_*' directories in {input_dir}.")
        return
    print(f"Found {len(case_dirs)} case directories.")

    evaluation_results = []

    for case_dir in tqdm(case_dirs, desc="Processing masks"):
        case_id = os.path.basename(case_dir)
        segmentations_dir = os.path.join(case_dir, 'segmentations')
        output_case_dir = os.path.join(output_dir, case_id)
        output_segmentations_dir = os.path.join(output_case_dir, 'segmentations')
        mask_files = glob.glob(os.path.join(segmentations_dir, 'vertebrae_*.nii.gz'))

        processed_mask_paths_by_name = {}
        first_affine = None
        first_shape = None

        for input_mask_path in tqdm(mask_files, desc=f"Processing {case_id} masks", leave=False):
            mask_filename = os.path.basename(input_mask_path)
            mask_name = mask_filename.replace('.nii.gz', '')
            output_mask_path = os.path.join(output_segmentations_dir, mask_filename)

            # post-processing on single mask
            results = postprocess_single_mask_gpu(input_mask_path, output_mask_path, device)

            if results[0] is not None:
                orig_comp, proc_comp, orig_vol, proc_vol = results
                evaluation_results.append({
                    'CaseID': case_id, 'MaskFile': mask_filename,
                    'OriginalComponents': orig_comp, 'ProcessedComponents': proc_comp,
                    'OriginalVolume': orig_vol, 'ProcessedVolume': proc_vol,
                    'VolumeChange': proc_vol - orig_vol, 'ComponentsRemoved': orig_comp - proc_comp
                })
                processed_mask_paths_by_name[mask_name] = output_mask_path
                if first_affine is None:
                    try:
                        nii_img = nib.load(output_mask_path)
                        first_affine = nii_img.affine
                        first_shape = nii_img.shape
                    except Exception as e:
                        print(f"Could not load processed mask {output_mask_path} to get affine/shape: {e}")

        # combine post-processed masks into combined_labels.nii.gz
        if processed_mask_paths_by_name and first_affine is not None and first_shape is not None:
            combined_label_map = np.zeros(first_shape, dtype=np.uint8)
            for label_value, mask_name in selected_class_map.items():
                if mask_name in processed_mask_paths_by_name:
                    processed_path = processed_mask_paths_by_name[mask_name]
                    try:
                        processed_nii = nib.load(processed_path)
                        processed_data = processed_nii.get_fdata().astype(bool)
                        combined_label_map[processed_data] = label_value
                    except Exception as e:
                        print(f"Error: Cannot load processed mask {processed_path} for combining (label {label_value}): {e}")

            combined_output_filename = "combined_labels.nii.gz"
            combined_output_path = os.path.join(output_case_dir, combined_output_filename)
            combined_nii = nib.Nifti1Image(combined_label_map, first_affine)
            try:
                os.makedirs(output_case_dir, exist_ok=True)
                nib.save(combined_nii, combined_output_path)
                print(f"Saved combined processed labels to: {combined_output_path}")
            except Exception as e:
                print(f"Error: Cannot save combined label file {combined_output_path}: {e}")


    # evaluation
    if evaluation_results:
        df = pd.DataFrame(evaluation_results)
        print("\n--- Post-processing Evaluation Summary ---")
        total_masks = len(df)
        masks_improved_connectivity = len(df[df['ComponentsRemoved'] > 0])
        avg_components_before = df['OriginalComponents'].mean()
        avg_components_after = df['ProcessedComponents'].mean()
        avg_volume_change = df['VolumeChange'].mean()
        print(f"Total number of processed vertebrae masks: {total_masks}")
        print(f"Number of masks with improved connectivity, i.e., reduced components: {masks_improved_connectivity} ({masks_improved_connectivity/total_masks:.2%})")
        print(f"Average number of components per mask before processing: {avg_components_before:.2f}")
        print(f"Average number of components per mask after processing: {avg_components_after:.2f}")
        print(f"Average volume change per mask before and after processing: {avg_volume_change:.2f} voxels")
        output_csv_path = os.path.join(output_dir, 'postprocessing_evaluation.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"\nEvaluation summary saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-processing predicted 3D vertebrae segmentation masks")
    parser.add_argument('--input_dir', type=str, required=True, help="directory of prediction results (e.g., ./AbdomenAtlasDemoPredict)")
    parser.add_argument('--output_dir', type=str, required=True, help="directory to save post-processed results (e.g., ./AbdomenAtlasDemoPredict_PostProcessed)")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)