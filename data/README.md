# Camera–LiDAR Sensor Fusion for 3D Vehicle Detection

A sensor-fusion pipeline that combines **YOLOv8 instance segmentation** on camera images with **Velodyne LiDAR point clouds** from the KITTI dataset to produce per-car 3D distance estimates — the foundation of every ADAS/ADS perception stack.

![Fused output sample](results/fused_006037.png)

**Author:** Aditya Athavale
**Course:** LiDAR and Radar Systems, M.Sc. Mechatronics, Hochschule Ravensburg-Weingarten (RWU)

---

## 1. Project Motivation

A modern autonomous vehicle can never rely on a single sensor. Each has a blind spot the others must cover:

- **LiDAR** provides accurate 3D geometry and depth, but produces a raw cloud of points with **no semantic meaning** — it doesn't know what a "car" is.
- **RGB cameras** provide rich colour, texture and object-class cues, but have **no reliable depth information** and degrade badly in poor lighting or occlusions.

This project implements and evaluates a **3D object-detection pipeline that fuses both modalities**, using **YOLOv8 instance segmentation** on the camera image and the corresponding **Velodyne LiDAR point cloud** from the KITTI dataset.

> The camera tells us *which pixels belong to a car*.
> The LiDAR tells us *how far each laser dot travelled*.
> Fusing them gives us *what* the object is **and** *how far* it is.

### 1.1 Why multimodal fusion?

Perception in safety-critical systems demands both accuracy and robustness. Single-modality systems have known failure modes:

- **LiDAR alone** — good depth, weak on class identity and fine object boundaries.
- **Camera alone** — rich semantics, fails on distance estimation and struggles in poor light or heavy occlusion.

A hybrid pipeline that combines YOLOv8 segmentation with 3D LiDAR geometry mirrors the sensor-fusion strategies used in real ADAS/ADS stacks.

### 1.2 Tools and dataset

| Tool | Role |
|------|------|
| **YOLOv8-seg (`yolov8m-seg.pt`)** | Instance segmentation to isolate car regions at pixel level |
| **KITTI 3D Object Detection** | Synchronised camera + LiDAR + calibration + ground-truth labels |
| **OpenCV / NumPy** | Image handling, matrix maths, fused visualisations |
| **Open3D** *(optional)* | Full 3D point-cloud visualisation and 3D-box inclusion tests |

---

## 2. Methodology

The pipeline in `src/sensor_fusion.py` runs in six clearly commented stages:

### Stage 1 — Data acquisition
For every 6-digit frame ID, build paths to the rectified image, `.bin` LiDAR scan, calibration `.txt`, and label `.txt`. Skip the frame if any input is missing.

### Stage 2 — Segmentation (camera path)
Run YOLOv8-seg on the image, keep only detections of COCO class 2 = *car*, resize each mask to the full 1242 × 375 resolution using nearest-neighbour interpolation (so binary edges stay sharp), and OR-combine them into a single **master car mask**.

### Stage 3 — Calibration and transformation (LiDAR path)
Load `P2`, `R0_rect` and `Tr_velo_to_cam`, then project every Velodyne point through the standard KITTI chain:

LiDAR (x, y, z) ──Tr_velo_to_cam──► camera 3D
──R0_rect────────► rectified camera 3D
──P2─────────────► (u·Z, v·Z, Z) = pixels + depth


Perspective divide (`u = u·Z / Z`) yields the 2D pixel of each 3D point, and any point outside the image rectangle is discarded.

### Stage 4 — Point-inclusion fusion
For every detected car instance, collect the projected LiDAR points whose `(u, v)` pixel falls inside that specific segmentation mask. That set is the **sub-pointcloud** for that car — the laser dots that physically hit that vehicle.

### Stage 5 — Metric calculation
For each car with a sub-pointcloud, take the **median depth** as the distance estimate. Median is preferred over mean because it is robust to a few *bleed-out* points that inevitably leak in from the background. Match the detection to the ground-truth car whose 2D box contains the detection's centroid, then log the absolute error in centimetres.

For a stricter, point-level evaluation, each LiDAR point can additionally be tested for **inclusion inside the transformed 3D bounding box** of a ground-truth car. Points inside count as **True Positives (TP)**, points outside count as **False Negatives (FN)**:

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)


### Stage 6 — Visualisation and output
LiDAR dots that landed on a detected car are drawn on the image, coloured on a **red = near → green = far** gradient (clipped at 40 m). GT boxes are drawn in green with the true distance labelled on top. Everything is saved to `results/`.

---

## 3. Use of Calibration Matrices

Aligning the LiDAR cloud with the image and with the ground-truth labels is the single most important step. Two matrices carry the heavy lifting.

### 3.1 Extrinsic matrix `Tr_velo_to_cam`

`Tr_velo_to_cam` transforms points from the **LiDAR** frame into the **camera** frame. In our pipeline it is used to project LiDAR points onto the image. When ground-truth 3D bounding boxes need to be tested against the LiDAR cloud, the **inverse** transform `T_cam→velo` is applied to the 8 corners of each GT box so the box sits in the same coordinate system as the point cloud:

```python
corners_hom  = np.hstack([corners, np.ones((8, 1))])
corners_velo = (T_cam_to_velo @ corners_hom.T).T[:, :3]
```

Getting this wrong is catastrophic: without it, ground-truth boxes do not overlap the LiDAR cloud at all, TP collapses to ~0, and every metric becomes meaningless.

### 3.2 Rectification matrix `R0_rect`

A 3 × 3 rotation that moves points from the raw camera frame into the **stereo-rectified** frame that KITTI's `P2` expects. Skipping this step produces a systematic geometric offset in the projection.

### 3.3 Intrinsic / projection matrix `P2`

The 3 × 4 projection matrix for the left colour camera. It maps a rectified 3D point to homogeneous pixel coordinates `(u·Z, v·Z, Z)`. The intrinsic block

K = [[fx, 0, cx],
[0, fy, cy],
[0, 0, 1 ]]


is embedded inside `P2` (together with a small stereo baseline term). `P2` is what makes the segmentation-mask filtering possible in the first place.

### 3.4 Homogeneous coordinates

All transforms are stacked using homogeneous 4D vectors `[x, y, z, 1]` so a rotation and a translation can be combined into a single matrix multiply. Standard practice in 3D graphics and computer vision; keeps the code short and vectorised.

---

## 4. Results

For every processed frame the pipeline emits two files into `results/`:

- `fused_<frame_id>.png` — the RGB image overlaid with:
  - **Green rectangles** = ground-truth 2D bounding boxes
  - **Green text** = ground-truth distance in metres
  - **Coloured dots** = LiDAR returns on detected cars (**red = near, green = far**, clipped at 40 m)
- `metrics_<frame_id>.txt` — a plain-text report with point counts per car, estimated vs. ground-truth distance, and absolute error in centimetres.

### 4.1 Interpreting the frames

Frames typically fall into four qualitative classes:

| Scene type | Behaviour |
|-----------|-----------|
| **High recall / clean** | Well-lit cars, dense point returns, TP fraction ≥ 15 %, near-zero distance error |
| **Sparse / low recall** | Distant or partially occluded cars, few LiDAR returns, high FN, recall < 3 % |
| **Moderate / mixed density** | Typical urban scene with varying vehicle distances — mid-range recall |
| **Dense / challenging** | Overlapping cars, clutter, diagonal orientations — bleed-out dominates, precision suffers |

### 4.2 Example metric interpretation

| Scenario | Distance error vs. ground truth |
|----------|-------------------------------|
| Car at ~12 m with 400+ returns | **10–30 cm** |
| Car beyond 25 m with few returns | Several metres |

Error at longer ranges is a direct consequence of LiDAR angular resolution — not a pipeline defect.

---

## 5. Discussion — Would we deploy this in a real car?

**No — not in this form.** The current implementation is a valuable prototype and teaching tool for sensor fusion, but it is missing several capabilities a production ADAS stack requires:

- **No real-time inference.** YOLOv8 medium runs comfortably on a desktop GPU, but neither the projection loop nor the per-point inclusion test is optimised for embedded targets.
- **No temporal tracking.** Every frame is treated independently; a Kalman filter or DeepSORT-style tracker is needed for object persistence across frames.
- **Cars only.** COCO class 2 is the only class kept. Pedestrians, cyclists, trucks etc. are ignored.
- **Occlusion handling is naive.** A single median depth per car breaks when the same YOLO mask covers two partially occluded cars.
- **False positives are assumed zero.** The precision figure is therefore optimistic.

The pipeline is well suited to studying calibration-error effects on fusion quality, comparing point-inclusion vs. bounding-box IoU as evaluation metrics, and serving as a base for a 3D bounding-box regressor or DBSCAN bleed-out filter.

---

## 6. Repository Structure

sensor-fusion-kitti/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│ └── sensor_fusion.py # main pipeline (6 stages)
│
├── data/
│ └── README.md # KITTI dataset download instructions
│
└── results/ # generated outputs
├── fused_<frame_id>.png
└── metrics_<frame_id>.txt


Each frame is identified by a 6-digit ID (e.g. `006037`) shared across all four KITTI data folders.

---

## 7. Dataset Setup

This project uses the **KITTI 3D Object Detection** benchmark. The dataset is publicly available but not included in this repository.

### Downloading

1. Register for a free account at [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
2. Download the following KITTI files:
   - **Left colour images** (`data_object_image_2.zip`, ~12 GB)
   - **Velodyne point clouds** (`data_object_velodyne.zip`, ~29 GB)
   - **Camera calibration matrices** (`data_object_calib.zip`, ~16 MB)
   - **Training labels** (`data_object_label_2.zip`, ~5 MB)
3. Unzip them into the `data/` folder preserving the KITTI structure (see [`data/README.md`](data/README.md) for the exact layout).
4. Update the `BASE_DIR` constant at the top of `src/sensor_fusion.py` to point at your local `data/` folder.

### Mini subset for testing

The pipeline was developed and validated on a 20-frame mini subset. Any contiguous range of KITTI training frames (6-digit IDs) can be used.

### License

The KITTI dataset is distributed under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)**. Please respect the original authors' licensing terms when using the data.

---

## 8. How to Run

### Requirements

```bash
pip install -r requirements.txt
pip install open3d  # optional, for full 3D visualisation and 3D-box inclusion tests
```

The first run automatically downloads `yolov8m-seg.pt` (~55 MB) from the Ultralytics servers.

### Running the pipeline

```bash
python src/sensor_fusion.py
```

A live window shows each fused frame in sequence. Press **`q`** to quit early. All outputs are saved in `results/`. If your directory layout differs, only the `BASE_DIR` constant at the top of `sensor_fusion.py` needs editing.

---

## 9. Future Work

- Replace the 2D-box GT match with a full **oriented 3D bounding-box point-inclusion test**, using `h, w, l, x, y, z, ry` from the label file and the inverse `Tr_velo_to_cam`.
- Reduce bleed-out by clustering each sub-pointcloud in 3D (**DBSCAN** or **Euclidean clustering**) and keeping only the dominant cluster before taking the median.
- Add **temporal tracking** (Kalman filter) so object identities persist across frames.
- Extend to more classes (pedestrian, cyclist, truck) and to the full **KITTI-360** dataset for a scale-up study.
- Add batched GPU inference so the pipeline can run at real-time frame rates.

---

## 10. References

1. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. [KITTI 3D Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
3. [KITTI-360 Dataset](http://www.cvlibs.net/datasets/kitti-360/)
4. [Open3D](http://www.open3d.org/docs/)
5. S. Elser, Lecture Notes, *LiDAR and Radar Systems*, Hochschule Ravensburg-Weingarten (RWU), 2024.

---

## Author

**Aditya Athavale** — M.Sc. Mechatronics, Ravensburg-Weingarten University of Applied Sciences

[GitHub](https://github.com/adityaathavale2) · [LinkedIn](https://www.linkedin.com/in/adityaathavale/)