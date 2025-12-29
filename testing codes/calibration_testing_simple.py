import cv2
import numpy as np
import json

# ================= LOAD CALIBRATION =================
CALIBRATION_FILE = "camera_calibration_aruco.json"
CAMERA_INDEX = 1
MARKER_LENGTH = 0.009  # 9mm

try:
    with open(CALIBRATION_FILE, 'r') as f:
        calib_data = json.load(f)
    
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['dist_coeffs'])
    
    print("✅ Calibration loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading calibration: {e}")
    exit()

# ================= SETUP =================
CAMERA_INDEX = 1
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# Global variables to store clicked points
points = []
frame_copy = None

def mouse_click(event, x, y, flags, param):
    """Handle mouse clicks"""
    global points, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"✓ Point {len(points)} clicked at ({x}, {y})")
        
        if len(points) == 2:
            # Calculate pixel distance
            p1, p2 = points[0], points[1]
            pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            # Convert to real-world distance using calibration
            # Using focal length from camera matrix
            fx = camera_matrix[0, 0]
            
            # Estimate real distance using marker size as reference
            # Real distance ≈ (pixel distance / focal length) * depth
            # For simplicity, assume marker size = MARKER_LENGTH at that distance
            real_distance_m = (pixel_dist / fx) * MARKER_LENGTH * 100
            real_distance_mm = real_distance_m * 1000
            real_distance_cm = real_distance_mm / 10
            
            print(f"\n📏 Distance Measurement:")
            print(f"   Pixel distance: {pixel_dist:.2f} px")
            print(f"   Real distance: {real_distance_cm:.2f} cm ({real_distance_mm:.1f} mm)")
            
            points = []  # Reset for next measurement

print("\n📸 Click Distance Measurement")
print("💡 Click 2 points to measure distance")
print("🔄 Press SPACE to clear points")
print("❌ Press ESC to exit\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        display = frame.copy()
        
        # Draw clicked points
        for i, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display, f"P{i+1}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw line between points if 2 are selected
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 0), 2)
            mid = ((points[0][0] + points[1][0]) // 2, 
                   (points[0][1] + points[1][1]) // 2)
            
            # Calculate distance
            pixel_dist = np.sqrt((points[0][0] - points[1][0])**2 + 
                               (points[0][1] - points[1][1])**2)
            fx = camera_matrix[0, 0]
            real_distance_m = (pixel_dist / fx) * MARKER_LENGTH * 100
            real_distance_cm = real_distance_m * 1000 / 10
            
            distance_text = f"Distance: {real_distance_cm:.2f} cm"
            cv2.putText(display, distance_text, mid, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Info text
        cv2.putText(display, f"Points: {len(points)}/2", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(display, "Click 2 points | SPACE: Clear | ESC: Exit", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        
        # Set mouse callback
        cv2.imshow("Distance Measurement", display)
        cv2.setMouseCallback("Distance Measurement", mouse_click)
        
        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            points = []
            print("🔄 Cleared points")
        elif key == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done")