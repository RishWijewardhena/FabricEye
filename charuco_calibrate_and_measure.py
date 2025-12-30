"""
Two tools in one file:

1) calibrate_and_save_extrinsics -- run to collect ChArUco calibration frames using a live camera,
   compute intrinsics and save camera_calibration.json, and (optionally) capture a single
   ChArUco placement in the scene to compute & save extrinsics (camera_extrinsics.json).

2) measure_with_fixed_camera -- interactive measurement mode that uses the saved intrinsics
   and extrinsics to convert clicked image points into metric coordinates on the board/plane
   and print distances.

USAGE (exact steps):
 - Run this script: python charuco_calibrate_and_measure.py
 - Follow on-screen instructions.

Requirements:
 - OpenCV >= 4.11.0
 - Numpy

Notes:
 - The camera must remain fixed after you compute and save extrinsics. If the camera moves,
   repeat the extrinsics capture step (press 'b').
 - The extrinsics (rvec,tvec) represent the pose of the ChArUco board in camera coordinates
   at the time you press the 'b' key to "capture board in scene". Place the board on the
   physical plane where you want to measure distances (or on a reference position whose
   transform to the measurement plane you know).

"""

import cv2
import numpy as np
import json
import os

# ---------- CONFIG ----------
CALIB_FILE = "camera_calibration.json"    # saved intrinsics
EXTRINSICS_FILE = "camera_extrinsics.json" # saved extrinsics (rvec,tvec) for the board pose in scene
CAMERA_INDEX = 1  # change if needed (0,1,2...)

# ChArUco board parameters (must match the physical board you use)
DICT_TYPE = cv2.aruco.DICT_5X5_50
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.012
MARKER_LENGTH = 0.009
MIN_CHARUCO_CORNERS = 4 

# ----------------------------

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# Build detectors and board
# ArUco dictionary describing marker bit patterns/ids (e.g. DICT_5X5_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)

# Create a ChArUco board object: (cols, rows), square side, marker side, and the ArUco dictionary
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

# Generic ArUco detector tuning parameters (adjust fields to tweak detection behavior)
params = cv2.aruco.DetectorParameters()

# ChArUco-specific detection/refinement parameters (separate in newer OpenCV versions)
charuco_params = cv2.aruco.CharucoParameters()

# Detector that handles ChArUco board detection using the board spec and params
charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, params)

# Detector for plain ArUco marker detection (used to find markers before ChArUco corner refinement)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)


def calibrate_camera_live():
    """Collect frames from live camera, detect ChArUco, perform calibration and save intrinsics.
    Controls while running:
      - SPACE: capture frame (if ChArUco board detected)
      - c    : run calibration when enough frames collected
      - q    : quit/cancel
    Output:
      - camera_calibration.json saved with camera_matrix and dist_coeffs
      - also saves calibration_rvecs_tvecs.json with per-frame poses (optional)
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    frame_count = 0

    print("\n=== Live ChArUco calibration ===")
    print("Instructions:")
    print(" - Show the board to the camera and move/rotate the board (camera fixed).")
    print(" - Press SPACE or left-click to capture a frame when the board is well-detected.")
    print(" - Collect 15-30 varied frames. Press 'c' to run calibration when ready.")
    print(" - Press 'q' to cancel.")

    #create a window for the mouse callback
    captured_click=False
    calibrate_click=False
    def _on_click(event, x, y, flags, param):
        nonlocal captured_click, calibrate_click #nonlocal to modify outer variable
        if event == cv2.EVENT_LBUTTONDOWN:
            captured_click = True

        if event==cv2.EVENT_RBUTTONDOWN:
            calibrate_click = True

    cv2.namedWindow('Calibration -Live')
    cv2.setMouseCallback('Calibration -Live', _on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        disp = frame.copy()
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(disp, marker_corners, marker_ids)
        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(disp, charuco_corners, charuco_ids)
            cv2.putText(disp, f"Charuco corners: {len(charuco_corners)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(disp, f"Captured frames: {frame_count}", (10, disp.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow('Calibration -Live', disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or captured_click: #check for space key or mouse click
            captured_click=False #reset for next click
            if charuco_ids is not None and len(charuco_ids) > MIN_CHARUCO_CORNERS:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                frame_count += 1
                print(f"Captured frame {frame_count} with {len(charuco_corners)} corners")
            else:
                print("Not enough Charuco corners detected — move board closer/rotate")

        elif key == ord('c') or calibrate_click: #check for 'c' key or right mouse click
            calibrate_click=False #reset for next click
            if frame_count < 8: #minimum frames to enable the checking
                print("Collect at least ~8-15 frames for stable intrinsics. Currently:", frame_count)
                continue
            print("Running calibration...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_charuco_corners, all_charuco_ids, board, image_size, None, None
            )
            if not ret:
                print("Calibration failed")
                continue
            print(f"Calibration done. RMS = {ret}")
            calib_data = {
                'camera_matrix': camera_matrix.tolist(),
                'dist_coeffs': dist_coeffs.tolist(),
                'rms': float(ret),
                'image_size': image_size
            }
            save_json(CALIB_FILE, calib_data)
            print(f"Saved intrinsics to {CALIB_FILE}")

            # Save per-frame poses optionally (convert to lists)
            poses = []
            for rv, tv in zip(rvecs, tvecs):
                poses.append({'rvec': rv.reshape(-1).tolist(), 'tvec': tv.reshape(-1).tolist()})
            save_json('calibration_rvecs_tvecs.json', poses)
            print("Saved calibration poses to calibration_rvecs_tvecs.json")
            break

        elif key == ord('q'):
            print("Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()


def capture_and_save_extrinsics():
    """Capture a single view of the board placed in the scene (the plane you want to measure on)
    and save rvec/tvec as extrinsics. Requires camera_calibration.json to exist.

    Controls while running:
      - SPACE: capture frame (board must be visible on measurement plane)
      - b    : compute extrinsics from that frame and save to camera_extrinsics.json
      - q    : quit
    """
    if not os.path.exists(CALIB_FILE):
        raise FileNotFoundError(f"Run intrinsics calibration first and save to {CALIB_FILE}")

    K, dist = load_calibration(CALIB_FILE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera to capture extrinsics")

    print("\n=== Capture board in scene to compute EXTRINSICS ===")
    print("Place the board on the plane where you will do measurements and press SPACE to capture.")
    print("Press 'b' to compute extrinsics from the captured frame and save. Press 'q' to cancel.")

    captured_img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        disp = frame.copy()
        cv2.imshow('Capture extrinsics - live', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured_img = frame.copy()
            print("Captured image for extrinsics computation — press 'b' to compute extrinsics")
            cv2.imshow('Capture extrinsics - live', captured_img)
        elif key == ord('b') and captured_img is not None:
            gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
            # detect
            marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
                print("Not enough charuco corners detected in captured image. Try again.")
                continue
            obj_pts = board.getChessboardCorners()[charuco_ids.flatten()]
            img_pts = charuco_corners.reshape(-1,2)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                print("solvePnP failed to compute extrinsics")
                continue
            save_json(EXTRINSICS_FILE, {'rvec': rvec.reshape(-1).tolist(), 'tvec': tvec.reshape(-1).tolist()})
            print(f"Saved extrinsics to {EXTRINSICS_FILE}")
            break
        elif key == ord('q'):
            print("Extrinsics capture cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()


# Small helper to load intrinsics file and return numpy arrays
def load_calibration(path):
    data = load_json(path)
    K = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['dist_coeffs'], dtype=np.float64)
    return K, dist


# ------------------------ Measurement tool ------------------------

def measurement_ui():
    """Interactive measurement using saved intrinsics and extrinsics.
    Controls:
      - SPACE : freeze a frame
      - click two points → distance computed automatically
      - q : quit
    """
    if not os.path.exists(CALIB_FILE):
        raise FileNotFoundError("Run calibration first and save intrinsics")

    K, dist = load_calibration(CALIB_FILE)

    rvec = tvec = R = plane_normal_cam = d = None

    if os.path.exists(EXTRINSICS_FILE):
        e = load_json(EXTRINSICS_FILE)
        rvec = np.array(e['rvec'], dtype=np.float64).reshape(3, 1)
        tvec = np.array(e['tvec'], dtype=np.float64).reshape(3,)
        R, _ = cv2.Rodrigues(rvec)
        plane_normal_cam = R[:, 2]
        d = -plane_normal_cam.dot(tvec)
    else:
        print("⚠ No extrinsics found — measurement will not work")

    display = None
    clicks = []
    measuring = False

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks, display
        if event == cv2.EVENT_LBUTTONDOWN and display is not None and not measuring:
            clicks.append((x, y))
            tmp = display.copy()
            for i, (ux, uy) in enumerate(clicks):
                cv2.circle(tmp, (ux, uy), 6, (0, 0, 255), -1)
                cv2.putText(
                    tmp, f"{i+1}", (ux+6, uy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2
                )
            cv2.imshow('Measure', tmp)

    cv2.namedWindow('Measure')
    cv2.setMouseCallback('Measure', on_mouse)

    print("\n=== Measurement UI ===")
    print("SPACE: freeze frame")
    print("Click two points → distance computed automatically")
    print("q: quit")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    frozen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if not frozen:
            cv2.imshow('Measure', frame)

        key = cv2.waitKey(10) & 0xFF

        # -------- Freeze frame --------
        if key == ord(' '):
            display = frame.copy()
            clicks = []
            frozen = True
            cv2.imshow('Measure', display)
            print("Frame frozen. Click two points.")

        # -------- Compute distance automatically --------
        if len(clicks) == 2 and frozen:
            if R is None:
                print("❌ No extrinsics available")
                clicks = []
                continue

            measuring = True
            try:
                p1 = image_to_plane(clicks[0], K, dist, R, tvec, plane_normal_cam, d)
                p2 = image_to_plane(clicks[1], K, dist, R, tvec, plane_normal_cam, d)
                dist_m = np.linalg.norm(p1 - p2)
            except Exception as e:
                print("Projection error:", e)
                clicks = []
                measuring = False
                continue

            print(f"Distance = {dist_m:.6f} m  ({dist_m*100:.2f} cm)")

            out = display.copy()
            cv2.line(out, clicks[0], clicks[1], (255, 0, 0), 2)
            mid = (
                (clicks[0][0] + clicks[1][0]) // 2,
                (clicks[0][1] + clicks[1][1]) // 2
            )
            cv2.putText(
                out, f"{dist_m*100:.2f} cm", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
            )

            cv2.imshow('Measure', out)

            clicks = []          # reset for next measurement
            measuring = False

        # -------- Quit --------
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Helper function: project image point to plane (board/object coordinates)
def image_to_plane(pt, K, dist, R, tvec, plane_normal_cam, d):
    # undistort to normalized coordinates
    pts = np.array(pt, dtype=np.float64).reshape(-1,1,2)
    und = cv2.undistortPoints(pts, K, dist, P=None)
    x = und[0,0,0]; y = und[0,0,1]
    ray_cam = np.array([x, y, 1.0])
    denom = plane_normal_cam.dot(ray_cam)
    if abs(denom) < 1e-9:
        raise RuntimeError('Ray nearly parallel to plane')
    s = -d / denom
    Xc = s * ray_cam
    obj_xy = R[:, :2].T.dot(Xc - tvec)
    return np.array([obj_xy[0], obj_xy[1], 0.0])


# ---------------------- MAIN MENU ----------------------
if __name__ == '__main__':
    print("\nCharuco calibration & measurement helper")
    print("Options:")
    print("1 - calibrate intrinsics (live camera)")
    print("2 - capture extrinsics (board in scene) and save")
    print("3 - run measurement UI (requires saved intrinsics + extrinsics)")
    print("q - quit")

    while True:
        c = input("Choose option (1/2/3/q): ").strip().lower()
        if c == '1':
            calibrate_camera_live()
        elif c == '2':
            capture_and_save_extrinsics()
        elif c == '3':
            measurement_ui()
        elif c == 'q':
            break
        else:
            print("Unknown option")

    print("Done.\n")
