"""
=====================================================================
 TASK 2 — SENSOR FUSION (Camera + LiDAR) on the KITTI mini-dataset
=====================================================================
 What this script does, in plain English:
 ---------------------------------------------------------------------
 The car has TWO "eyes":
   1. A COLOUR CAMERA  -> good at recognising WHAT an object is
                          (car, person, tree...) but bad at knowing
                          HOW FAR AWAY it is.
   2. A LIDAR SENSOR   -> shoots laser dots at the world and knows
                          exactly HOW FAR each dot travelled. But
                          the raw dots are just a cloud of numbers,
                          they do not "know" what a car is.

 The idea of SENSOR FUSION is to combine both:
   Camera says   -> "here are the pixels that belong to a car"
   LiDAR says    -> "these laser dots landed on those pixels, and
                    they came back after 12.4 metres"
   Together      -> "there is a car 12.4 metres in front of us"

 The 6 STAGES of this script:
   STAGE 1  Load one KITTI frame (image + LiDAR + calibration + labels)
   STAGE 2  CAMERA path : run YOLOv8 segmentation to find cars in pixels
   STAGE 3  LIDAR  path : project the 3D laser dots onto the 2D image
   STAGE 4  FUSION      : keep only the laser dots that landed on a car
   STAGE 5  DISTANCE    : take the median depth of those dots = car range
   STAGE 6  EVALUATION  : compare our distance to the ground-truth label

 Run it and you will get, for every frame:
   fused_XXXXXX.png    -> picture with GT boxes + coloured LiDAR dots
   metrics_XXXXXX.txt  -> text log with distance error per car
=====================================================================
"""

# ---------------------------------------------------------------------
# IMPORTS - the tools we need
# ---------------------------------------------------------------------
import cv2                          # OpenCV  -> read images, draw shapes, show windows
import numpy as np                  # NumPy   -> matrix / array maths
import os                           # OS      -> build file paths, check files exist
import glob                         # glob    -> list files matching a pattern
from ultralytics import YOLO        # YOLOv8  -> the neural network that finds cars


# =====================================================================
# STAGE 0 — CONFIGURATION  (edit these paths for your machine)
# =====================================================================
# The KITTI images are always this size (fixed by the dataset).
IMG_WIDTH  = 1242
IMG_HEIGHT = 375

# Root folder of the KITTI mini-dataset on your PC.
BASE_DIR = r"C:\Desktop\MASTERS MECHATRONICS\SEM 2\LIDAR AND RADAR SYSTEMS\Compulsory project\TASK(PROJECT)2\KITTI-Selection_incl_LiDAR_2026"

# Sub-folders inside the dataset. The names come from KITTI.
CALIB_DIR    = os.path.join(BASE_DIR, r"data_object_calib\training\calib")
IMAGE_DIR    = os.path.join(BASE_DIR, r"data_object_image_2     (left top camera)\training\image_2")
LABEL_DIR    = os.path.join(BASE_DIR, r"data_object_label_2     (3D BB data)\training\label_2")
VELODYNE_DIR = os.path.join(BASE_DIR, r"data_object_velodyne     (3D point cloud lidar data)\training\velodyne")

# Where we save the results (fused pictures + text logs).
OUTPUT_DIR   = os.path.join(BASE_DIR, "fusion_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)   # create it if it does not exist yet

# Load the YOLO model ONCE (loading it inside the loop would be slow).
# "yolov8m-seg.pt" = medium size + segmentation (gives per-pixel masks).
model = YOLO("yolov8m-seg.pt")

# Find every image and grab just the 6-digit frame ID (e.g. "006037").
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
frame_ids   = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

print("=" * 60)
print("Initialisation OK")
print(f"Saving results to : {OUTPUT_DIR}")
print(f"Frames found      : {len(frame_ids)}")
print("=" * 60, "\n")


# =====================================================================
# HELPER — read a KITTI calibration .txt file
# =====================================================================
# KITTI stores its calibration matrices in a plain text file, one per
# line, e.g.  "P2: 7.21e+02 0.0 6.09e+02 ... "
# We split by ":" and turn the numbers into a NumPy array.
def read_kitti_calib(filepath):
    calib = {}
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():                # skip empty lines
                continue
            key, value = line.split(":")
            calib[key.strip()] = np.array(
                [float(x) for x in value.strip().split()]
            )
    return calib


# =====================================================================
# MAIN LOOP — process every frame, one at a time
# =====================================================================
for frame_id in frame_ids:
    print("\n" + "=" * 50)
    print(f" FRAME {frame_id}")
    print("=" * 50)

    # -----------------------------------------------------------------
    # STAGE 1 — build the file paths for this frame
    # -----------------------------------------------------------------
    image_path = os.path.join(IMAGE_DIR,    f"{frame_id}.png")
    lidar_path = os.path.join(VELODYNE_DIR, f"{frame_id}.bin")
    calib_path = os.path.join(CALIB_DIR,    f"{frame_id}.txt")
    label_path = os.path.join(LABEL_DIR,    f"{frame_id}.txt")

    # Where we will save the outputs for this frame.
    save_img_path = os.path.join(OUTPUT_DIR, f"fused_{frame_id}.png")
    save_log_path = os.path.join(OUTPUT_DIR, f"metrics_{frame_id}.txt")

    # If any input file is missing, skip this frame.
    if not (os.path.exists(image_path) and
            os.path.exists(lidar_path) and
            os.path.exists(calib_path)):
        print(f"  Missing data for frame {frame_id} — skipping.")
        continue

    # Open the log file. Everything we write() inside this block goes
    # into metrics_XXXXXX.txt.
    with open(save_log_path, "w") as log_file:
        log_file.write(f"=== SENSOR FUSION REPORT — FRAME {frame_id} ===\n\n")

        # =============================================================
        # STAGE 2 — CAMERA PATH  (find the cars in the image)
        # =============================================================
        # 2a. Read the RGB image from disk.
        img = cv2.imread(image_path)

        # 2b. A blank black canvas the same size as the image. Every
        #     pixel that YOLO says "is a car" will be set to 1 here.
        master_car_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        # 2c. Run YOLOv8. verbose=False keeps the console clean.
        #     [0] because we only sent 1 image.
        results = model(img, verbose=False)[0]

        # 2d. Loop through every object YOLO detected.
        if results.masks is not None:
            for box, mask in zip(results.boxes, results.masks):
                class_id = int(box.cls[0].item())

                # COCO class 2 = "car". Ignore people, trucks, bikes...
                if class_id == 2:
                    # The mask comes out at the model's own resolution,
                    # so we resize it back up to the full image size.
                    single_mask = mask.data[0].cpu().numpy()
                    single_mask = cv2.resize(single_mask,
                                             (IMG_WIDTH, IMG_HEIGHT),
                                             interpolation=cv2.INTER_NEAREST)

                    # Threshold at 0.5 -> hard 0/1 mask.
                    binary_mask = (single_mask > 0.5).astype(np.uint8)

                    # OR-combine every car into the master mask.
                    master_car_mask = cv2.bitwise_or(master_car_mask,
                                                    binary_mask)

        # =============================================================
        # STAGE 3 — LIDAR PATH  (project 3D dots onto the 2D image)
        # =============================================================
        # The full transformation chain is:
        #
        #    LiDAR (x,y,z)  --Tr_velo_to_cam-->  camera 3D
        #                   --R0_rect        -->  rectified camera 3D
        #                   --P2             -->  pixel (u,v)  + depth
        #
        # 3a. Load the three calibration matrices for this frame.
        m = read_kitti_calib(calib_path)
        P2             = m["P2"].reshape(3, 4)              # camera intrinsics
        R0_rect        = m["R0_rect"].reshape(3, 3)         # rectification
        Tr_velo_to_cam = m["Tr_velo_to_cam"].reshape(3, 4)  # LiDAR -> camera

        # 3b. Read the raw LiDAR .bin file. Each point = 4 float32
        #     values : (x, y, z, intensity).
        raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # Keep only points in FRONT of the sensor (x > 0). Drop intensity.
        points_3d = raw[raw[:, 0] > 0, :3]
        n = points_3d.shape[0]

        # 3c. Add a "1" column so we can do rotation+translation together
        #     with a single matrix multiply. This is called "homogeneous
        #     coordinates" and is a standard trick in 3D graphics.
        pts_h = np.hstack((points_3d, np.ones((n, 1))))          # (N, 4)

        # 3d. LiDAR frame  ->  camera frame
        pts_cam  = Tr_velo_to_cam @ pts_h.T                       # (3, N)

        # 3e. camera frame  ->  rectified camera frame
        pts_rect = R0_rect @ pts_cam                              # (3, N)

        # 3f. rectified 3D  ->  pixel coordinates via P2.
        #     Again we add a row of 1s to make it homogeneous.
        pts_rect_h = np.vstack((pts_rect, np.ones((1, pts_rect.shape[1]))))
        pts_pix_h  = P2 @ pts_rect_h                              # (3, N)

        # 3g. "Perspective divide": divide u and v by depth (z) so that
        #     things far away appear smaller — like your eyes do.
        u_scaled, v_scaled, depth = pts_pix_h[0], pts_pix_h[1], pts_pix_h[2]
        u_pix = (u_scaled / depth).astype(np.int32)
        v_pix = (v_scaled / depth).astype(np.int32)

        # 3h. Keep only points that landed INSIDE the image rectangle.
        inside = (u_pix >= 0) & (u_pix < IMG_WIDTH) & \
                 (v_pix >= 0) & (v_pix < IMG_HEIGHT)

        # Final list : each row = [pixel_x, pixel_y, distance_in_metres]
        projected_lidar = np.column_stack((u_pix[inside],
                                           v_pix[inside],
                                           depth[inside]))

        # =============================================================
        # STAGE 4 — FUSION  +  read the ground-truth labels
        # =============================================================
        # A copy of the image we will draw on. We keep the original clean.
        fused = img.copy()

        # 4a. Read ground-truth labels (one line per object in the scene).
        ground_truth_cars = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    tok = line.split()
                    if tok[0] != "Car":         # only care about cars
                        continue
                    xmin = int(float(tok[4]));  ymin = int(float(tok[5]))
                    xmax = int(float(tok[6]));  ymax = int(float(tok[7]))
                    true_depth = float(tok[13])   # column 13 = z (metres)

                    ground_truth_cars.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                        "true_depth": true_depth,
                    })

                    # Draw the GT box in GREEN + the true distance.
                    cv2.rectangle(fused, (xmin, ymin), (xmax, ymax),
                                  (0, 255, 0), 2)
                    cv2.putText(fused, f"GT: {true_depth:.2f}m",
                                (xmin, ymin - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 1)

        # 4b. For EACH detected car, build its own "sub-pointcloud".
        #     A sub-pointcloud is the set of LiDAR dots that landed
        #     inside that specific car's segmentation mask.
        sub_pointclouds = {}
        if results.masks is not None:
            for idx, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                if int(box.cls[0].item()) != 2:
                    continue

                # per-instance binary mask, resized to full image size
                s = mask.data[0].cpu().numpy()
                s = cv2.resize(s, (IMG_WIDTH, IMG_HEIGHT),
                               interpolation=cv2.INTER_NEAREST)
                inst_mask = (s > 0.5).astype(np.uint8)

                # Check for each projected LiDAR dot: is its pixel = 1
                # in this car's mask? If yes, it belongs to this car.
                u = projected_lidar[:, 0].astype(np.int32)
                v = projected_lidar[:, 1].astype(np.int32)
                inside_this_car = inst_mask[v, u] == 1

                sub_pointclouds[idx] = projected_lidar[inside_this_car]
                log_file.write(
                    f"Car #{idx}: sub-pointcloud has "
                    f"{len(sub_pointclouds[idx])} LiDAR points.\n"
                )

        # =============================================================
        # STAGE 5 — DISTANCE per car  +  compare to ground truth
        # =============================================================
        log_file.write("\n--- Distance evaluation ---\n")
        print("Metrics:")

        for idx, pts in sub_pointclouds.items():
            if len(pts) == 0:               # nothing landed on that car
                continue

            # Use the MEDIAN depth. Median is robust against a few
            # noisy "bleed-out" points that might have leaked in from
            # the background. Mean would be pulled off by outliers.
            estimated_distance = np.median(pts[:, 2])

            # Where is this car (in pixels) on average?
            avg_u = np.mean(pts[:, 0])
            avg_v = np.mean(pts[:, 1])

            # Find the ground-truth car whose 2D box contains this centroid.
            matched_gt = None
            for gt in ground_truth_cars:
                x1, y1, x2, y2 = gt["bbox"]
                if x1 <= avg_u <= x2 and y1 <= avg_v <= y2:
                    matched_gt = gt
                    break

            if matched_gt is not None:
                true_distance   = matched_gt["true_depth"]
                absolute_error  = abs(estimated_distance - true_distance)
                msg = (f"Car #{idx}  ->  MATCH  | "
                       f"Estimated: {estimated_distance:.2f} m | "
                       f"Ground truth: {true_distance:.2f} m | "
                       f"Error: {absolute_error * 100:.1f} cm\n")
            else:
                msg = (f"Car #{idx}  ->  no matching GT | "
                       f"Estimated: {estimated_distance:.2f} m\n")

            print(" " + msg.strip())
            log_file.write(msg)

        # =============================================================
        # STAGE 6 — draw the fused LiDAR dots on the image
        # =============================================================
        # We only draw the dots that fell INSIDE a car mask.
        # Colour goes from RED (near, <2m) to GREEN (far, >=40m).
        for i in range(len(projected_lidar)):
            u = int(projected_lidar[i, 0])
            v = int(projected_lidar[i, 1])
            d = projected_lidar[i, 2]

            if master_car_mask[v, u] == 1:
                norm_d = min(d, 40.0) / 40.0                 # 0..1
                colour = (0, int(255 * norm_d), int(255 * (1.0 - norm_d)))
                cv2.circle(fused, (u, v), 2, colour, -1)

        # -----------------------------------------------------------------
        # Save the outputs for this frame
        # -----------------------------------------------------------------
        cv2.imwrite(save_img_path, fused)
        print(f" [SAVED] {save_img_path}")
        print(f" [SAVED] {save_log_path}")

    # -----------------------------------------------------------------
    # Live preview window — press "q" to quit early
    # -----------------------------------------------------------------
    cv2.imshow("Sensor Fusion — press 'q' to quit", fused)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        print("Loop broken by user.")
        break

cv2.destroyAllWindows()
print("\n" + "=" * 50)
print(f" DONE. All outputs saved to: {OUTPUT_DIR}")
print("=" * 50)
