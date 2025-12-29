# # import cv2
# # import numpy as np

# # # Camera
# # CAMERA_INDEX = 2

# # # ArUco dictionaries
# # ARUCO_DICTS = {
# #     "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
# #     "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
# #     "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
# #     "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
# #     "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
# #     "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
# #     "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
# #     "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
# # }

# # cap = cv2.VideoCapture(CAMERA_INDEX)
# # if not cap.isOpened():
# #     print("❌ Could not open camera")
# #     exit()

# # print("Press 'q' to quit.")

# # try:
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             continue

# #         detected_any = False
# #         try:
# #             for name, dict_id in ARUCO_DICTS.items():
# #                 # Create dictionary and detector
# #                 aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
# #                 detector = cv2.aruco.ArucoDetector(aruco_dict)

# #                 # Detect markers
# #                 corners, ids, rejected = detector.detectMarkers(frame)

# #                 if ids is not None and len(ids) > 0:
# #                     detected_any = True
# #                     print(f"✅ Detected markers with dictionary: {name}")
# #                     cv2.aruco.drawDetectedMarkers(frame, corners, ids)
# #                     break  # Stop at first working dictionary
# #         except Exception as e:
# #             print(f"⚠️ Error detecting markers: {e}")
# #             continue

# #         cv2.imshow("ArUco Detection", frame)

# #         key = cv2.waitKey(1) & 0xFF
# #         if key == ord('q'):
# #             break
# # finally:
# #     cap.release()
# #     cv2.destroyAllWindows()


# import cv2
# print(cv2.__version__)

# import cv2
# import numpy as np

# # Use calibration results
# camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float64)
# dist_coeffs = np.array(calibration_data["dist_coeffs"], dtype=np.float64)

# # 3D board points (same as used in calibration)
# board_obj_pts = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float32)

# # Capture a new image
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# ret, frame = cap.read()
# cap.release()

# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # Detect board
# marker_corners, marker_ids, charuco_corners, charuco_ids = charuco_detector.detectBoard(gray)

# if charuco_ids is None or len(charuco_ids) < 6:
#     raise RuntimeError("Not enough ChArUco corners detected")

# # Pair points safely
# ids = charuco_ids.flatten()
# img_pts = []
# obj_pts = []

# for i, cid in enumerate(ids):
#     if cid < len(board_obj_pts):
#         pt = charuco_corners[i][0]  # (x,y)
#         img_pts.append(pt)
#         obj_pts.append(board_obj_pts[cid])

# img_pts = np.array(img_pts, dtype=np.float32)
# obj_pts = np.array(obj_pts, dtype=np.float32)

# # Pose estimation
# success, rvec, tvec = cv2.solvePnP(
#     obj_pts,
#     img_pts,
#     camera_matrix,
#     dist_coeffs,
#     flags=cv2.SOLVEPNP_ITERATIVE
# )

# if not success:
#     raise RuntimeError("Pose estimation failed")

# # True distance (slanted camera)
# distance = np.linalg.norm(tvec)

# print(f"Camera → board distance: {distance*100:.2f} cm")
# print(f"Translation vector (meters): {tvec.ravel()}")
# print(f"Rotation vector (radians): {rvec.ravel()}")


# import cv2

# img = cv2.imread("test_image_charuco.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# dicts = [
#     cv2.aruco.DICT_4X4_50,
#     cv2.aruco.DICT_5X5_50,
#     cv2.aruco.DICT_6X6_50,
#     cv2.aruco.DICT_7X7_50,
# ]

# for d in dicts:
#     dictionary = cv2.aruco.getPredefinedDictionary(d)
#     parameters = cv2.aruco.DetectorParameters()
#     detector = cv2.aruco.ArucoDetector(dictionary, parameters)

#     corners, ids, rejected = detector.detectMarkers(gray)

#     print(f"{d}: {0 if ids is None else len(ids)} markers detected")

import cv2 as cv
# print(cv.getBuildInformation())
print("OpenCV version:", cv.__version__)
