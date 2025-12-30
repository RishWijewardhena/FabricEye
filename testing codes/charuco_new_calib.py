import numpy as np
import cv2
import json
import os

# ChArUco board configuration
DICT_TYPE = cv2.aruco.DICT_5X5_50
SQUARES_X = 5  # Number of squares in X direction
SQUARES_Y = 7  # Number of squares in Y direction
SQUARE_LENGTH = 0.012  # Square side length in meters (12mm)
MARKER_LENGTH = 0.009  # Marker side length in meters (9mm)

def calibrate_camera():
    """
    Calibrate camera using ChArUco board from webcam (camera index 2) update this as needed.
    """
    # Initialize ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )
    
    # Create ChArUco detector
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    
    # Storage for calibration
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    
    # Open webcam
    cap = cv2.VideoCapture(2) # Changed to camera index 2
    
    if not cap.isOpened():
        print("Error: Could not open camera 2. Trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera")
            return
    
    print("=== ChArUco Camera Calibration ===")
    print(f"Board: {SQUARES_X}x{SQUARES_Y} squares")
    print(f"Dictionary: DICT_5X5_50")
    print("\nInstructions:")
    print("- Move the ChArUco board to different positions and angles")
    print("- Press SPACE to capture a frame when board is detected")
    print("- Capture at least 15-20 good frames from different angles")
    print("- Press 'c' to calculate calibration")
    print("- Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        # Draw detected markers and corners
        display_frame = frame.copy()
        
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)
            
            if charuco_corners is not None and len(charuco_corners) > 3:
                cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)
                
                # Show status
                status = f"Detected: {len(charuco_corners)} corners | Frames: {frame_count}"
                cv2.putText(display_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Board not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display_frame, f"Captured: {frame_count}", (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            if charuco_corners is not None and len(charuco_corners) > 3:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                frame_count += 1
                print(f"Frame {frame_count} captured with {len(charuco_corners)} corners")
            else:
                print("Not enough corners detected. Try again.")
        
        elif key == ord('c'):  # Calculate calibration
            if frame_count >= 10:
                print(f"\nCalculating calibration with {frame_count} frames...")
                break
            else:
                print(f"Need at least 10 frames. Current: {frame_count}")
        
        elif key == ord('q'):  # Quit
            print("Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Perform calibration
    print("Running calibration...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None
    )
    
    if ret:
        print(f"\n=== Calibration Successful ===")
        print(f"RMS Error: {ret:.4f}")
        print(f"\nCamera Matrix:\n{camera_matrix}")
        print(f"\nDistortion Coefficients:\n{dist_coeffs}")
        
        # Save calibration data
        calibration_data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'rms_error': ret,
            'image_size': image_size,
            'square_length': SQUARE_LENGTH,
            'marker_length': MARKER_LENGTH
        }
        
        with open('camera_calibration.json', 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print("\nCalibration data saved to 'camera_calibration.json'")
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(all_charuco_corners)):
            projected, _ = cv2.projectPoints(
                board.getChessboardCorners()[all_charuco_ids[i]],
                rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )
            error = cv2.norm(all_charuco_corners[i], projected, cv2.NORM_L2) / len(projected)
            mean_error += error
        
        mean_error /= len(all_charuco_corners)
        print(f"Mean reprojection error: {mean_error:.4f} pixels")
    else:
        print("Calibration failed!")

if __name__ == "__main__":
    calibrate_camera()