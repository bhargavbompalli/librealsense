#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from helper_functions import (
    cv_find_chessboard,
    get_chessboard_points_3D,
    get_depth_at_pixel,
    convert_depth_pixel_to_metric_coordinate,
    convert_depth_frame_to_pointcloud,
    get_boundary_corners_2D,
    get_clipped_pointcloud
)
from calibration_kabsch import PoseEstimation, Transformation

def main():
    # Chessboard parameters
    chessboard_size = (3, 5)  # Number of inner corners per a chessboard row and column
    square_size = 0.010  # Size of a square in meters

    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 15)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    profile = pipeline.start(config)

    # Get camera intrinsics
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    # Prepare frames for calibration
    frames = {}
    frames[('device', 'product_line')] = {}
    frames[('device', 'product_line')][rs.stream.depth] = None
    frames[('device', 'product_line')][(rs.stream.infrared, 1)] = None

    try:
        # Wait for frames to stabilize
        for _ in range(30):
            pipeline.wait_for_frames()

        print("Starting camera calibration...")
        calibrated = False
        while not calibrated:
            frameset = pipeline.wait_for_frames()
            depth_frame = frameset.get_depth_frame()
            infrared_frame = frameset.get_infrared_frame(1)

            if not depth_frame or not infrared_frame:
                continue

            # Update frames for PoseEstimation
            frames[('device', 'product_line')][rs.stream.depth] = depth_frame
            frames[('device', 'product_line')][(rs.stream.infrared, 1)] = infrared_frame

            # Perform pose estimation using the helper class
            pose_estimator = PoseEstimation(frames, {'device': {rs.stream.depth: depth_intrinsics}}, [chessboard_size[0], chessboard_size[1], square_size])
            transformation_result = pose_estimator.perform_pose_estimation()

            # Check if calibration was successful
            calibrated = transformation_result['device'][0]
            if calibrated:
                transformation = transformation_result['device'][1]
                print("Camera calibrated successfully using Kabsch algorithm.")
            else:
                print("Calibration failed. Ensure the chessboard is fully visible and try again.")
                # Display infrared image for debugging
                infrared_image = np.asanyarray(infrared_frame.get_data())
                cv2.imshow('Calibration', infrared_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

        print("Capturing frames for measurement...")
        # Capture aligned frames
        align = rs.align(rs.stream.depth)
        frameset = pipeline.wait_for_frames()
        aligned_frames = align.process(frameset)
        depth_frame = aligned_frames.get_depth_frame()
        infrared_frame = aligned_frames.get_infrared_frame(1)
        color_frame = None  # Not using color frame in this case

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

        # Isolate cavity using helper function
        print("Isolating mold cavity...")
        points_array = np.asarray(remaining_cloud.points).T
        distances = a * points_array[0, :] + b * points_array[1, :] + c * points_array[2, :] + d
        cavity_indices = np.where(distances < -0.001)[0]  # Adjust threshold as needed
        cavity_cloud = remaining_cloud.select_by_index(cavity_indices)

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
        volume = mesh.get_volume()
        print(f"Estimated cavity volume: {volume:.6f} cubic meters")

        # Visualize result
        print("Visualizing cavity mesh...")
        o3d.visualization.draw_geometries([mesh],
            window_name='Mold Cavity Mesh',
            width=800, height=600)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
