# measure_mold_volume.py

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import json  # For loading calibration data
from helper_functions import convert_depth_frame_to_pointcloud
from calibration_kabsch import Transformation

def main():
    # Load calibration data
    try:
        with open('calibration_data.json', 'r') as f:
            calibration_data = json.load(f)
        print("Calibration data loaded from 'calibration_data.json'")
    except FileNotFoundError:
        print("Calibration data not found. Please run 'calibrate_camera.py' first.")
        return

    # Retrieve transformation matrix and depth intrinsics
    transformation_matrix_list = calibration_data['transformation_matrix']
    intrinsics_data = calibration_data['depth_intrinsics']

    # Reconstruct the transformation object
    transformation_matrix = np.array(transformation_matrix_list)
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]
    transformation = Transformation(rotation_matrix, translation_vector)

    # Reconstruct depth intrinsics
    depth_intrinsics = rs.intrinsics()
    depth_intrinsics.width = intrinsics_data['width']
    depth_intrinsics.height = intrinsics_data['height']
    depth_intrinsics.ppx = intrinsics_data['ppx']
    depth_intrinsics.ppy = intrinsics_data['ppy']
    depth_intrinsics.fx = intrinsics_data['fx']
    depth_intrinsics.fy = intrinsics_data['fy']
    depth_intrinsics.model = intrinsics_data['model']
    depth_intrinsics.coeffs = intrinsics_data['coeffs']

    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, depth_intrinsics.width, depth_intrinsics.height, rs.format.z16, 15)
    profile = pipeline.start(config)

    try:
        # Wait for frames to stabilize
        for _ in range(30):
            pipeline.wait_for_frames()

        input("Calibration loaded. Place the mold under the camera and press Enter to start measurement...")

        # Capture frames for measurement
        print("Capturing frames for measurement...")
        frameset = pipeline.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        if not depth_frame:
            print("Failed to capture depth frame.")
            return

        # Generate point cloud using helper function
        print("Generating point cloud...")
        depth_image = np.asanyarray(depth_frame.get_data())
        x, y, z = convert_depth_frame_to_pointcloud(depth_image, depth_intrinsics)
        points = np.vstack((x, y, z))

        # Apply transformation to align with the reference coordinate system
        print("Transforming point cloud to reference coordinate system...")
        points_transformed = transformation.inverse().apply_transformation(points)

        # Convert to Open3D point cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_transformed.T)

        # Segment the reference plane (top edges of the mold)
        print("Segmenting reference plane...")
        plane_model, inliers = pc.segment_plane(
            distance_threshold=0.005,
            ransac_n=3,
            num_iterations=1000)
        [a, b, c, d] = plane_model
        plane_cloud = pc.select_by_index(inliers)
        remaining_cloud = pc.select_by_index(inliers, invert=True)

        print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # Isolate cavity
        print("Isolating mold cavity...")
        points_array = np.asarray(remaining_cloud.points).T
        distances = a * points_array[0, :] + b * points_array[1, :] + c * points_array[2, :] + d
        cavity_indices = np.where(distances < -0.001)[0]  # Adjust threshold as needed
        cavity_cloud = remaining_cloud.select_by_index(cavity_indices)

        if len(cavity_cloud.points) == 0:
            print("No cavity points detected. Please ensure the mold is properly placed.")
            return

        # Compute volume
        print("Computing cavity volume...")
        cavity_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01, max_nn=30))

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cavity_cloud, depth=9)

        # Crop the mesh
        bbox = cavity_cloud.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

        if not mesh.is_watertight():
            print("The mesh is not watertight. Cannot compute volume.")
            return

        volume = mesh.get_volume()
        print(f"Estimated cavity volume: {volume:.6f} cubic meters")

        # Visualize result
        print("Visualizing cavity mesh...")
        o3d.visualization.draw_geometries([mesh],
            window_name='Mold Cavity Mesh',
            width=800, height=600)

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
