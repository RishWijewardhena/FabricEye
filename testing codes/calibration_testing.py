import cv2
import numpy as np
import json

# ================= LOAD CALIBRATION =================
CALIBRATION_FILE = "camera_calibration_aruco.json"
CAMERA_INDEX = 1
MARKER_LENGTH = 0.009  # 9mm - must match your calibration

# Load calibration data from JSON
try:
    with open(CALIBRATION_FILE, 'r') as f:
        calib_data = json.load(f)
    
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['dist_coeffs'])
    
    print("✅ Calibration loaded successfully")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs}")
    
except Exception as e:
    print(f"❌ Error loading calibration: {e}")
    exit()

# ================= SETUP DETECTOR =================
DICT_TYPE = cv2.aruco.DICT_5X5_50
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ================= CAMERA SETUP =================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("\n📸 Real-time Distance Measurement")
print("💡 Show 2 known markers to measure distance between them")
print("❌ Press ESC to exit\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Display frame
        display = frame.copy()
        
        # Draw detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            
            # ===== CALCULATE DISTANCES =====
            if len(ids) >= 2:
                # Get the first two markers
                marker1_corners = corners[0][0]  # 4 corners of first marker
                marker2_corners = corners[1][0]  # 4 corners of second marker
                
                # Calculate center of each marker (average of 4 corners)
                marker1_center = np.mean(marker1_corners, axis=0)
                marker2_center = np.mean(marker2_corners, axis=0)
                
                # Calculate pixel distance
                pixel_distance = np.linalg.norm(marker1_center - marker2_center)
                
                # ===== UNDISTORT & CALCULATE REAL DISTANCE =====
                # Undistort the corner points
                marker1_undist = cv2.undistortPoints(
                    marker1_corners.reshape(1, -1, 2),
                    camera_matrix,
                    dist_coeffs,
                    P=camera_matrix
                )
                
                marker2_undist = cv2.undistortPoints(
                    marker2_corners.reshape(1, -1, 2),
                    camera_matrix,
                    dist_coeffs,
                    P=camera_matrix
                )
                
                # Get undistorted centers
                m1_center_undist = np.mean(marker1_undist[0], axis=0)
                m2_center_undist = np.mean(marker2_undist[0], axis=0)
                
                # Extract focal length from camera matrix
                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                
                # Pixel distance in undistorted coordinates
                pixel_dist_undist = np.linalg.norm(m1_center_undist - m2_center_undist)
                
                # Estimate distance using marker size as reference
                # Since we know each marker is MARKER_LENGTH meters
                marker1_pixel_size = np.linalg.norm(marker1_corners[0] - marker1_corners[1])
                
                # Real distance = (known marker size) * (pixel distance) / (marker pixel size)
                if marker1_pixel_size > 0:
                    scale = MARKER_LENGTH / marker1_pixel_size
                    real_distance = pixel_distance * scale
                else:
                    real_distance = 0
                
                # Display information
                cv2.line(display, 
                        tuple(marker1_center.astype(int)), 
                        tuple(marker2_center.astype(int)), 
                        (0, 255, 0), 2)
                
                mid_point = ((marker1_center + marker2_center) / 2).astype(int)
                
                # Show distance in millimeters and centimeters
                distance_mm = real_distance * 1000
                distance_cm = distance_mm / 10
                
                text = f"Distance: {distance_cm:.2f} cm ({distance_mm:.1f} mm)"
                cv2.putText(display, text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show marker IDs
                ids_text = f"Markers: ID {ids[0][0]} & ID {ids[1][0]}"
                cv2.putText(display, ids_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
            elif len(ids) == 1:
                # Show single marker info
                cv2.putText(display, "Show 2 markers to measure distance", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(display, f"Detected 1 marker (ID: {ids[0][0]})", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(display, "No markers detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show number of detected markers
        num_markers = len(ids) if ids is not None else 0
        cv2.putText(display, f"Detected markers: {num_markers}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        
        cv2.imshow("Distance Measurement", display)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Measurement test completed")