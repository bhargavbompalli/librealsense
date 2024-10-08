#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

def calibrate_camera():
    """
    Calibrate the camera using a chessboard pattern to obtain intrinsic parameters.
    """
    # Chessboard parameters
    chessboard_size = (3, 5)  # Inner corners per a chessboard row and column
    square_size = 0.010  # Size of a square in your defined unit (meters)

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on the real chessboard dimensions
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    pipeline.start(config)

    try:
        print("Starting camera calibration...")
        collected = 0
        while collected < 20:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)
                corners_subpix = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_subpix)

                # Draw and display the corners
                cv2.drawChessboardCorners(
                    color_image, chessboard_size, corners_subpix, ret)
                collected += 1
                print(f"Calibration images collected: {collected}/20")
                cv2.imshow('Calibration', color_image)
                cv2.waitKey(500)
            else:
                cv2.imshow('Calibration', color_image)
                cv2.waitKey(1)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # Camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("Camera calibrated successfully.")
        return camera_matrix, dist_coeffs
    else:
        raise Exception("Camera calibration failed.")

def capture_frames(pipeline, align):
    """
    Capture aligned depth and color frames.
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    return depth_frame, color_frame

def create_point_cloud(depth_frame, color_frame, intrinsics):
    """
    Generate a point cloud from depth and color frames.
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Get dimensions
    h, w = depth_image.shape

    # Create point cloud
    pc = o3d.geometry.PointCloud()
    # Intrinsics
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    # Create grid of coordinates
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth_image * depth_frame.get_units()
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    # Stack to get point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0

    # Remove zero depth points
    mask = z.reshape(-1) > 0
    points = points[mask]
    colors = colors[mask]

    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    return pc

def segment_plane(pc, distance_threshold=0.005, ransac_n=3, num_iterations=1000):
    """
    Segment the largest plane in the point cloud.
    """
    plane_model, inliers = pc.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)
    plane_cloud = pc.select_by_index(inliers)
    remaining_cloud = pc.select_by_index(inliers, invert=True)
    return plane_model, plane_cloud, remaining_cloud

def isolate_cavity(pc, plane_model, threshold=-0.001):
    """
    Isolate points below the reference plane (mold cavity).
    """
    [a, b, c, d] = plane_model
    points = np.asarray(pc.points)
    distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    cavity_indices = np.where(distances < threshold)[0]
    cavity_cloud = pc.select_by_index(cavity_indices)
    return cavity_cloud

def compute_cavity_volume(cavity_cloud, poisson_depth=9):
    """
    Compute the volume of the cavity using Poisson reconstruction.
    """
    cavity_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.01, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cavity_cloud, depth=poisson_depth)
    bbox = cavity_cloud.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    volume = mesh.get_volume()
    return volume, mesh

def main():
    # Calibrate camera
    try:
        camera_matrix, dist_coeffs = calibrate_camera()
    except Exception as e:
        print(f"Camera calibration failed: {e}")
        return

    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    
    try:
        profile = pipeline.start(config)
    except rs.error as e:
        print(f"Failed to start pipeline: {e}")
        return

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Get camera intrinsics
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    try:
        print("Capturing frames...")
        depth_frame, color_frame = None, None
        for _ in range(30):  # Try for 30 frames
            depth_frame, color_frame = capture_frames(pipeline, align)
            if depth_frame is not None and color_frame is not None:
                break
        
        if depth_frame is None or color_frame is None:
            raise RuntimeError("Failed to capture valid frames after 30 attempts")

        print("Generating point cloud...")
        pc = create_point_cloud(depth_frame, color_frame, intrinsics)

        print("Segmenting reference plane...")
        plane_model, plane_cloud, remaining_cloud = segment_plane(pc)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        print("Isolating mold cavity...")
        cavity_cloud = isolate_cavity(remaining_cloud, plane_model)

        print("Computing cavity volume...")
        volume, mesh = compute_cavity_volume(cavity_cloud)
        print(f"Estimated cavity volume: {volume:.6f} cubic meters")

        print("Visualizing cavity mesh...")
        o3d.visualization.draw_geometries([mesh],
            window_name='Mold Cavity Mesh',
            width=800, height=600)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()