# calibrate_camera.py

import pyrealsense2 as rs
import numpy as np
import cv2
import json  # Use JSON for serialization
from helper_functions import cv_find_chessboard, get_chessboard_points_3D
from calibration_kabsch import PoseEstimation

def main():
    # Chessboard parameters
    chessboard_size = (3, 5)  # Inner corners per a chessboard row and column
    square_size = 0.025  # Size of a square in meters

    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 15)
    profile = pipeline.start(config)

    # Get camera intrinsics
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

                # Extract transformation matrix
                transformation_matrix = transformation.pose_mat.tolist()  # Convert to list for JSON serialization

                # Extract intrinsics data
                intrinsics_data = {
                    'width': depth_intrinsics.width,
                    'height': depth_intrinsics.height,
                    'ppx': depth_intrinsics.ppx,
                    'ppy': depth_intrinsics.ppy,
                    'fx': depth_intrinsics.fx,
                    'fy': depth_intrinsics.fy,
                    'model': depth_intrinsics.model,
                    'coeffs': depth_intrinsics.coeffs
                }

                # Save calibration data as JSON
                calibration_data = {
                    'transformation_matrix': transformation_matrix,
                    'depth_intrinsics': intrinsics_data
                }

                with open('calibration_data.json', 'w') as f:
                    json.dump(calibration_data, f)
                print("Calibration data saved to 'calibration_data.json'")
                break
            else:
                print("Calibration failed. Ensure the chessboard is fully visible and try again.")
                # Display infrared image for debugging
                infrared_image = np.asanyarray(infrared_frame.get_data())
                cv2.imshow('Calibration', infrared_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
