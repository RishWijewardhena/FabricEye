import numpy as np
import cv2
import json
import os

# Global variables for mouse callback
points_2d = []
points_3d = []
drawing = False

def load_calibration():
    """Load camera calibration data"""
    if not os.path.exists('camera_calibration.json'):
        print("Error: camera_calibration.json not found!")
        print("Please run calibration first.")
        return None, None
    
    with open('camera_calibration.json', 'r') as f:
        data = json.load(f)
    
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])
    
    print("Calibration data loaded successfully")
    print(f"RMS Error: {data['rms_error']:.4f}")
    
    return camera_matrix, dist_coeffs

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for selecting points"""
    global points_2d, drawing
    
    frame, camera_matrix, dist_coeffs = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_2d) < 2:
            points_2d.append((x, y))
            print(f"Point {len(points_2d)} selected: ({x}, {y})")
            
            # Draw point
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"P{len(points_2d)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if len(points_2d) == 2:
                # Draw line between points
                cv2.line(frame, points_2d[0], points_2d[1], (255, 0, 0), 2)
                drawing = True

def get_pose_from_aruco(frame, camera_matrix, dist_coeffs, marker_length=0.05):
    """
    Detect ArUco markers and get pose estimation
    marker_length in meters (default 50mm)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    rvecs, tvecs = None, None
    
    if ids is not None and len(ids) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate pose for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )
        
        # Draw axis for each marker
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                            rvecs[i], tvecs[i], marker_length * 0.5)
    
    return rvecs, tvecs, ids

def pixel_to_3d(point_2d, camera_matrix, dist_coeffs, rvec, tvec, z_plane=0):
    """
    Convert 2D pixel coordinates to 3D world coordinates
    Assumes point lies on a plane at z_plane distance in marker frame
    """
    # Undistort point
    point_2d = np.array([[point_2d]], dtype=np.float32)
    undistorted = cv2.undistortPoints(point_2d, camera_matrix, dist_coeffs, P=camera_matrix)
    
    # Convert to normalized camera coordinates
    x_norm = (undistorted[0][0][0] - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y_norm = (undistorted[0][0][1] - camera_matrix[1, 2]) / camera_matrix[1, 1]
    
    # Get rotation and translation
    R, _ = cv2.Rodrigues(rvec)
    
    # Ray direction in camera frame
    ray_camera = np.array([x_norm, y_norm, 1.0])
    
    # Transform to world frame
    ray_world = R.T @ ray_camera
    camera_pos = -R.T @ tvec.flatten()
    
    # Intersect with plane at z = z_plane
    if abs(ray_world[2]) < 1e-6:
        return None
    
    t = (z_plane - camera_pos[2]) / ray_world[2]
    point_3d = camera_pos + t * ray_world
    
    return point_3d

def calculate_distance_simple(point1_2d, point2_2d, camera_matrix, dist_coeffs, depth):
    """
    Simple distance calculation assuming both points are at same depth
    depth in meters
    """
    # Undistort points
    points = np.array([[point1_2d, point2_2d]], dtype=np.float32)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    
    # Convert to normalized image coordinates
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Calculate 3D coordinates assuming same depth
    x1 = (undistorted[0][0][0] - cx) * depth / fx
    y1 = (undistorted[0][0][1] - cy) * depth / fy
    z1 = depth
    
    x2 = (undistorted[0][1][0] - cx) * depth / fx
    y2 = (undistorted[0][1][1] - cy) * depth / fy
    z2 = depth
    
    # Calculate Euclidean distance
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    return distance

def measure_distance():
    """
    Main function to measure distance between two points
    """
    global points_2d, points_3d, drawing
    
    # Load calibration
    camera_matrix, dist_coeffs = load_calibration()
    if camera_matrix is None:
        return
    
    # Open webcam
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
    
    print("\n=== Distance Measurement ===")
    print("\nMethods:")
    print("1. With ArUco marker: Place a marker of known size in the scene")
    print("2. Manual depth: Enter approximate depth to object")
    print("\nInstructions:")
    print("- Click two points on a known object")
    print("- Press ENTER to calculate distance")
    print("- Press 'r' to reset points")
    print("- Press 'q' to quit\n")
    
    # Ask for marker size or depth
    print("Enter reference marker size in mm (or press Enter to use manual depth): ")
    marker_input = input().strip()
    
    use_aruco = False
    marker_length = 0.05  # Default 50mm
    manual_depth = 0.5  # Default 500mm
    
    if marker_input:
        try:
            marker_length = float(marker_input) / 1000.0  # Convert mm to meters
            use_aruco = True
            print(f"Using ArUco marker of size {marker_input}mm")
        except:
            print("Invalid input, using manual depth")
    
    if not use_aruco:
        print("Enter approximate depth to object in mm (default 500mm): ")
        depth_input = input().strip()
        if depth_input:
            try:
                manual_depth = float(depth_input) / 1000.0
            except:
                pass
        print(f"Using manual depth: {manual_depth*1000}mm")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # Try to detect ArUco markers if using that method
        rvec, tvec, marker_ids = None, None, None
        if use_aruco:
            rvec, tvec, marker_ids = get_pose_from_aruco(
                display_frame, camera_matrix, dist_coeffs, marker_length
            )
            
            if marker_ids is not None:
                cv2.putText(display_frame, f"Marker detected: {len(marker_ids)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No marker detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw selected points
        for i, pt in enumerate(points_2d):
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{i+1}", (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(points_2d) == 2:
            cv2.line(display_frame, points_2d[0], points_2d[1], (255, 0, 0), 2)
        
        # Instructions
        cv2.putText(display_frame, "Click two points, then press ENTER", 
                   (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Distance Measurement', display_frame)
        cv2.setMouseCallback('Distance Measurement', mouse_callback, 
                            (display_frame, camera_matrix, dist_coeffs))
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if len(points_2d) == 2:
                # Calculate distance
                if use_aruco and rvec is not None and tvec is not None:
                    # Use ArUco pose estimation
                    p1_3d = pixel_to_3d(points_2d[0], camera_matrix, dist_coeffs, 
                                       rvec[0], tvec[0])
                    p2_3d = pixel_to_3d(points_2d[1], camera_matrix, dist_coeffs, 
                                       rvec[0], tvec[0])
                    
                    if p1_3d is not None and p2_3d is not None:
                        distance = np.linalg.norm(p2_3d - p1_3d)
                        print(f"\n=== Distance (ArUco-based) ===")
                        print(f"Point 1: {p1_3d}")
                        print(f"Point 2: {p2_3d}")
                        print(f"Distance: {distance*1000:.2f} mm ({distance*10:.2f} cm)")
                else:
                    # Use manual depth
                    distance = calculate_distance_simple(
                        points_2d[0], points_2d[1], 
                        camera_matrix, dist_coeffs, manual_depth
                    )
                    print(f"\n=== Distance (Depth-based) ===")
                    print(f"Assumed depth: {manual_depth*1000:.2f} mm")
                    print(f"Distance: {distance*1000:.2f} mm ({distance*10:.2f} cm)")
                
                print(f"Pixel distance: {np.linalg.norm(np.array(points_2d[1]) - np.array(points_2d[0])):.2f} pixels")
                print("\nPress 'r' to measure again\n")
            else:
                print("Please select 2 points first")
        
        elif key == ord('r'):  # Reset
            points_2d = []
            points_3d = []
            drawing = False
            print("Points reset")
        
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    measure_distance()