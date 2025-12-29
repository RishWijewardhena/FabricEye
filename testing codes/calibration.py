import cv2
import numpy as np

# ================= CONFIG =================
CAMERA_INDEX = 1              # external webcam
DICT_TYPE = cv2.aruco.DICT_5X5_50 #5x5 dictionary with 50 unique markers
MARKER_LENGTH = 0.009         # meters (9 mm) — based on your board
MARKER_SEPARATION = 0.0015     # meters (1 mm) — adjust if needed
MARKERS_X = 5
MARKERS_Y = 9                 # your board has 9 rows
NUM_IMAGES_REQUIRED = 20
# SAVE_FILE = "camera_calibration_aruco.npz"

# ================= CREATE BOARD =================
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
board = cv2.aruco.GridBoard(
    (MARKERS_X, MARKERS_Y),
    MARKER_LENGTH,
    MARKER_SEPARATION,
    aruco_dict
)
detector = cv2.aruco.ArucoDetector(aruco_dict) #creates a detector object that will find markers in camera images

# ================= CAPTURE =================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera")

all_corners = []
all_ids = []
image_size = None

print("📸 Press SPACE to capture image")
print("❌ Press ESC to finish capturing")

try:
    while True:
        ret, frame = cap.read() #get BGR image from camera
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray) #feed the grayscale image to the detector to find markers

        display = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids) #draw detected markers on the original BGR image for visualization

        cv2.putText(display, f"Captured: {len(all_ids)} / {NUM_IMAGES_REQUIRED}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("ArUco Calibration", display)

        key = cv2.waitKey(1)

        if key == 32 and ids is not None:  #  User pressed SPACE key (ASCII 32)
            all_corners.append(corners)
            all_ids.append(ids)
            image_size = gray.shape[::-1]
            print(f"✔ Captured image {len(all_ids)}")

        elif key == 27:  # ESC
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# ================= CALIBRATION =================
if len(all_ids) < 5: #Checks if you captured at least 5 images
    raise RuntimeError("❌ Not enough calibration images")

print("🔧 Calibrating camera...")

try:
    # For OpenCV 4.11.0: Manually create object points based on board parameters
    object_points = []
    image_points = []
    
    # Board parameters
    marker_ids_on_board = np.arange(MARKERS_X * MARKERS_Y).reshape(MARKERS_Y, MARKERS_X)
    
    for image_idx, detected_ids in enumerate(all_ids):
        '''
        image_idx = Image number (0, 1, 2, ...)
        detected_ids = IDs of markers detected in that image
        '''
        detected_corners = all_corners[image_idx]
        
        if detected_ids is None or len(detected_ids) == 0:
            continue #Stop executing the remaining code in this loop and jump to the next iteration.
        
        # For each detected marker
        for detection_idx, marker_id in enumerate(detected_ids.flatten()):
            # Find the position of this marker on the board
            board_pos = np.where(marker_ids_on_board.flatten() == marker_id)[0] #flatten() converts 2D array to 1D
            #Finds which position on the board this marker ID

            if len(board_pos) > 0:
                pos_index = board_pos[0]
                marker_row = pos_index // MARKERS_X
                marker_col = pos_index % MARKERS_X
                
                # Calculate 3D object points for this marker's corners
                marker_x = marker_col * (MARKER_LENGTH + MARKER_SEPARATION)
                marker_y = marker_row * (MARKER_LENGTH + MARKER_SEPARATION)
                
                # Four corners of the marker in 3D space
                obj_pts = np.array([
                    [marker_x, marker_y, 0],
                    [marker_x + MARKER_LENGTH, marker_y, 0],
                    [marker_x + MARKER_LENGTH, marker_y + MARKER_LENGTH, 0],
                    [marker_x, marker_y + MARKER_LENGTH, 0]
                ], dtype=np.float32) #dtype specifies the data type of the array elements
                
                #All have Z=0 because markers are on a flat board
                object_points.append(obj_pts)
                
                # Image points (detected corners)
                img_pts = detected_corners[detection_idx].astype(np.float32)
                if img_pts.ndim == 3:
                    img_pts = img_pts.reshape(-1, 2)
                image_points.append(img_pts)
    
    if len(object_points) == 0:
        raise RuntimeError("❌ No valid marker correspondences found")
    
    print(f"📊 Using {len(object_points)} marker sets for calibration")
    
    # Calibrate camera with CALIB_USE_LU flag for faster computation
    print("⏳ Running calibration algorithm (this may take a moment)...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, # 3D points in real world space (meter units)
        image_points, # Where those points appear in camera (pixels)
        image_size, #Image resolution
        None,
        None,
        flags=cv2.CALIB_USE_LU  # Faster computation using LU decomposition
    )
    
    print("✅ Calibration complete")
    
except Exception as e:
    print(f"⚠️ Calibration error: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n✅ Calibration complete")
print("Reprojection error:", ret)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# ================= SAVE =================
# import os

# # Validate save path
# save_dir = os.path.dirname(SAVE_FILE) or "."
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir, exist_ok=True)

# try:
#     np.savez(
#         SAVE_FILE,
#         camera_matrix=camera_matrix,
#         dist_coeffs=dist_coeffs
#     )
#     print(f"\n💾 Calibration saved to {SAVE_FILE}")
# except Exception as e:
#     print(f"\n❌ Error saving calibration: {e}")

import os 
import json

SAVE_FILE_JSON = "camera_calibration_aruco.json"
# Validate save path    
save_dir = os.path.dirname(SAVE_FILE_JSON) or "."

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

try:
    #convert numpy arrays to lists for JSON serialization
    calibration_data={
        "camera_matrix":camera_matrix.tolist(),
        "dist_coeffs":dist_coeffs.flatten().tolist(),
        "reprojection_error":float(ret),
                "image_size": image_size,
        "config": {
            "marker_length": MARKER_LENGTH,
            "marker_separation": MARKER_SEPARATION,
            "markers_x": MARKERS_X,
            "markers_y": MARKERS_Y,
            "num_images": len(all_ids)
        }
    }

    with open(SAVE_FILE_JSON, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    print(f"\n💾 Calibration saved to {SAVE_FILE_JSON}")

except Exception as e:
    print(f"\n❌ Error saving calibration: {e}")