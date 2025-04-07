# Visualization for vertebrae segmentation masks
# Author: Zhilin Zhang

# Requires:
# !pip install pyvista

import os
import nibabel as nib
import numpy as np
import pyvista as pv
import argparse
import random

try:
    UniformGrid = pv.UniformGrid
except AttributeError:
    from pyvista.core import UniformGrid

if os.environ.get('DISPLAY', '') == '':
    try:
        pv.start_xvfb()
    except Exception as e:
        print("Failed to start xvfb:", e)

def load_nii_to_mesh(nii_path, threshold=0.5):
    img = nib.load(nii_path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]  # (sx, sy, sz)
    dims = data.shape  # (nx, ny, nz)
    grid = UniformGrid()
    grid.dimensions = dims
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    flat_data = data.ravel(order='F')
    grid.point_data["values"] = flat_data
    mesh = grid.threshold(threshold, scalars="values")
    return mesh

def add_meshes_from_combined(subject_path, plotter):
    combined_path = os.path.join(subject_path, "combined_labels.nii.gz")
    if not os.path.exists(combined_path):
        return False
    print(f"Loading combined_labels: {combined_path}")
    img = nib.load(combined_path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    dims = data.shape
    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels > 0]
    for label in unique_labels:
        mask = (data == label).astype(np.float32)
        grid = UniformGrid()
        grid.dimensions = dims
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = mask.ravel(order='F')
        mesh = grid.threshold(0.5, scalars="values")
        color = [random.random() for _ in range(3)]
        plotter.add_mesh(mesh, color=color, opacity=1.0, smooth_shading=True, label=f"Label {int(label)}")
    return True

def add_meshes_from_segmentations(subject_path, plotter):
    segmentation_dir = os.path.join(subject_path, "segmentations")
    if not os.path.exists(segmentation_dir):
        print(f"[Error] Segmentation folder not found: {segmentation_dir}")
        return
    nii_files = [f for f in os.listdir(segmentation_dir)
                 if f.startswith("vertebrae_") and f.endswith(".nii.gz")]
    def sort_key(fname):
        base = fname.replace("vertebrae_", "").replace(".nii.gz", "")
        letter = base[0]
        number = int(base[1:])
        order = {"C": 0, "T": 1, "L": 2}
        return (order.get(letter, 3), number)
    nii_files = sorted(nii_files, key=sort_key)
    for nii_file in nii_files:
        nii_path = os.path.join(segmentation_dir, nii_file)
        print(f"Loading file: {nii_path}")
        mesh = load_nii_to_mesh(nii_path, threshold=0.5)
        color = [random.random() for _ in range(3)]
        plotter.add_mesh(mesh, color=color, opacity=1.0, smooth_shading=True, label=nii_file)

# window_size=(2560,1440)
def capture_views(plotter, output_prefix, window_size=(2560,1440)):
    # high-resolution window
    plotter.window_size = window_size

    bounds = plotter.renderer.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
    offset = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 1.5

    # camera parameters for three views: (position, focal_point, view_up)
    views = {
        "lateral": (
            (bounds[1] + offset, center[1], center[2] + offset/2), tuple(center), (0, 0, 1)
        ),
        "anterior": (
            (center[0], bounds[3] + offset, center[2]), tuple(center), (0, 0, 1)
        ),
        "posterior": (
            (center[0], bounds[2] - offset, center[2]), tuple(center), (0, 0, 1)
        )
    }

    for view_name, (position, focal_point, view_up) in views.items():
        plotter.camera_position = (position, focal_point, view_up)
        plotter.render()
        filename = f"{output_prefix}_{view_name}.png"
        plotter.screenshot(filename)

def visualize_subject(subject_path, screenshot_prefix="compare"):
    # off_screen mode for faster rendering
    plotter = pv.Plotter(off_screen=True)
    
    if not add_meshes_from_combined(subject_path, plotter):
        add_meshes_from_segmentations(subject_path, plotter)
    
    plotter.add_legend(bcolor="w", face=None, loc="upper right")
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.render()
    capture_views(plotter, screenshot_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D visualization of vertebral segmentation results")
    parser.add_argument("--predict_dir", type=str, default="AbdomenAtlasDemoPredict_PostProcessed", help="root directory for prediction results ")
    parser.add_argument("--subject", type=str, default="BDMAP_00000006", help="subject folder name, e.g., BDMAP_00000006 or BDMAP_00000031")
    parser.add_argument("--screenshot_prefix", type=str, default="post-processing", help="prefix for output screenshot files")
    args = parser.parse_args()
    
    subject_path = os.path.join(args.predict_dir, args.subject)
    visualize_subject(subject_path, screenshot_prefix=args.screenshot_prefix)