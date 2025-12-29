import cv2
import numpy as np
import json
import os
import traceback

# ================= CONFIG =================
CAMERA_INDEX = 1
DICT_TYPE = cv2.aruco.DICT_5X5_50

SQUARE_LENGTH = 0.0125   # meters
MARKER_LENGTH = 0.009    # meters

MARKERS_X = 5
MARKERS_Y = 9

NUM_IMAGES_REQUIRED = 20
SAVE_FILE = "camera_calibration_charuco.json"

# ================= HEADER =================
print("=" * 60)
print("ChArUco Camera Calibration (robust pairing, works with opencv-python)")
print("=" * 60)
print(f"Board: {MARKERS_X} x {MARKERS_Y}")
print(f"Square: {SQUARE_LENGTH*1000:.1f} mm")
print(f"Marker: {MARKER_LENGTH*1000:.1f} mm")
print("=" * 60)

# ================= CREATE BOARD =================
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)

charuco_board = cv2.aruco.CharucoBoard(
    (MARKERS_X, MARKERS_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

charuco_detector = cv2.aruco.CharucoDetector(charuco_board)

# Precompute board 3D points (for indexing by charuco id)
board_obj_pts = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float32)  # shape (M,3)

# ================= CAPTURE =================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera")

all_object_points = []  # list of arrays shaped (N,3)
all_image_points = []   # list of arrays shaped (N,2)
image_size = None

print("\nSPACE = capture | ESC = finish\n")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectBoard returns: markerCorners, markerIds, charucoCorners, charucoIds
        marker_corners, marker_ids, charuco_corners, charuco_ids = charuco_detector.detectBoard(gray)

        display = frame.copy()
        count = 0
        if charuco_ids is not None:
            # Some returns have shape (N,1); flatten for safe count
            try:
                count = int(np.asarray(charuco_ids).flatten().shape[0])
            except Exception:
                count = 0

        cv2.putText(display, f"Captured: {len(all_object_points)}/{NUM_IMAGES_REQUIRED}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(display, f"ChArUco ids: {count}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("ChArUco Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 32 and count >= 6:  # SPACE and at least 6 corners
            # Normalize ids
            try:
                ids = np.asarray(charuco_ids, dtype=np.int32).flatten()
            except Exception:
                print("⚠️ Could not convert charuco_ids to array — skipping frame")
                continue

            # Build a normalized list of corner points indexed to ids.
            # charuco_corners may have shapes like (N,1,2), (N,2), a nested list, etc.
            corners_src = charuco_corners

            # Defensive conversion: try to index per id
            corners_list = []
            try:
                # If corners_src is None, skip
                if corners_src is None:
                    print("⚠️ charuco_corners is None — skipping frame")
                    continue

                # If it's an ndarray
                if isinstance(corners_src, np.ndarray):
                    # possible shapes: (N,1,2) or (N,2) or (N,4,2) (rare)
                    if corners_src.ndim == 3 and corners_src.shape[1] == 1:
                        # (N,1,2) -> get [i,0]
                        for i in range(corners_src.shape[0]):
                            pt = corners_src[i, 0]
                            corners_list.append((float(pt[0]), float(pt[1])))
                    elif corners_src.ndim == 3 and corners_src.shape[2] == 2 and corners_src.shape[1] != 1:
                        # e.g. (N,K,2) where K>1; take the first entry per detected corner
                        for i in range(corners_src.shape[0]):
                            pt = corners_src[i, 0]
                            corners_list.append((float(pt[0]), float(pt[1])))
                    elif corners_src.ndim == 2 and corners_src.shape[1] == 2:
                        # (N,2)
                        for i in range(corners_src.shape[0]):
                            pt = corners_src[i]
                            corners_list.append((float(pt[0]), float(pt[1])))
                    else:
                        # Fallback: flatten and chunk by 2
                        flat = np.asarray(corners_src, dtype=np.float32).reshape(-1)
                        if flat.size % 2 == 0:
                            pts = flat.reshape(-1, 2)
                            for p in pts:
                                corners_list.append((float(p[0]), float(p[1])))
                        else:
                            raise ValueError("Unexpected corner array shape")
                else:
                    # If not ndarray, try treating as iterable of points
                    for entry in corners_src:
                        # entry may be [ [x,y] ] or [x,y]
                        arr = np.asarray(entry, dtype=np.float32).reshape(-1)
                        if arr.size >= 2:
                            corners_list.append((float(arr[0]), float(arr[1])))
                        else:
                            raise ValueError("Unexpected corner entry shape")
            except Exception as e:
                print(f"⚠️ Error normalizing corners: {e}; skipping frame")
                continue

            # Now corners_list should be length >= ids length in most cases; but we only pair first len(ids) elements
            if len(corners_list) < ids.shape[0]:
                print(f"⚠️ Not enough corners ({len(corners_list)}) for ids ({ids.shape[0]}) — skipping frame")
                continue

            # Build paired lists using exact one-to-one mapping
            paired_obj = []
            paired_img = []
            for idx_in_list, cid in enumerate(ids):
                try:
                    cid_int = int(cid)
                except Exception:
                    print(f"⚠️ Invalid charuco id {cid} — skipping")
                    continue

                if not (0 <= cid_int < len(board_obj_pts)):
                    print(f"⚠️ charuco id {cid_int} out of range — ignoring")
                    continue

                # Use the corner at the same index (charuco convention)
                corner_xy = corners_list[idx_in_list]
                paired_obj.append(board_obj_pts[cid_int])
                paired_img.append([corner_xy[0], corner_xy[1]])

            # Final safety
            if len(paired_obj) != len(paired_img):
                print(f"⚠️ Paired count mismatch after pairing: {len(paired_obj)} vs {len(paired_img)} — skipping")
                continue

            if len(paired_obj) < 6:
                print(f"⚠️ Too few paired points ({len(paired_obj)}) — skipping")
                continue

            obj_arr = np.asarray(paired_obj, dtype=np.float32).reshape(-1, 3)  # (N,3)
            img_arr = np.asarray(paired_img, dtype=np.float32).reshape(-1, 2)  # (N,2)

            all_object_points.append(obj_arr)
            all_image_points.append(img_arr)
            image_size = gray.shape[::-1]  # (width, height)

            print(f"✔ Captured image {len(all_object_points)}: object pts = {obj_arr.shape[0]}, image pts = {img_arr.shape[0]}")

            if len(all_object_points) >= NUM_IMAGES_REQUIRED:
                print("✅ Required number of images captured")
                break

        elif key == 27:
            print("User requested exit")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# ================= VALIDATION =================
print(f"\n📊 Total good images collected: {len(all_object_points)}")
if len(all_object_points) < 3:
    raise RuntimeError(f"❌ Not enough valid images for calibration (need >=3). Got {len(all_object_points)}")

print("\nPer-image point counts (object_pts, image_pts):")
for i, (op, ip) in enumerate(zip(all_object_points, all_image_points)):
    print(f" Image {i}: {op.shape[0]} , {ip.shape[0]}")
    if op.shape[0] != ip.shape[0]:
        raise RuntimeError(f"❌ MISMATCH at image {i}: {op.shape[0]} object pts != {ip.shape[0]} image pts")

# ================= CALIBRATION =================
print("\n🔧 Running cv2.calibrateCamera ...")
try:
    flags = 0
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_object_points,
        imagePoints=all_image_points,
        imageSize=tuple(map(int, image_size)),
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags
    )
    print("✅ Calibration completed")
except Exception:
    print("⚠️ Calibration failed:")
    traceback.print_exc()
    raise

# ================= RESULTS =================
print("\n" + "=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)
print(f"Reprojection error: {ret:.6f}")
print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion coeffs:\n", dist_coeffs.flatten())
print("=" * 60)

# ================= SAVE =================
print(f"\n💾 Saving calibration to {SAVE_FILE} ...")
save_dir = os.path.dirname(SAVE_FILE) or "."
os.makedirs(save_dir, exist_ok=True)

calibration_data = {
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.flatten().tolist(),
    "reprojection_error": float(ret),
    "image_size": list(image_size),
    "config": {
        "board_type": "ChArUco",
        "markers_x": MARKERS_X,
        "markers_y": MARKERS_Y,
        "square_length": SQUARE_LENGTH,
        "marker_length": MARKER_LENGTH,
        "images_used": len(all_object_points)
    }
}

with open(SAVE_FILE, "w") as f:
    json.dump(calibration_data, f, indent=4)

print("✅ Calibration saved")
print("\n🎉 DONE")
