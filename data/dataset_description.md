# KITTI Dataset Subsample
This dataset is a subsample of the [KTIIT 3D Object Detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). It contains images, pointclouds and ground truth 3D bounding boxes for every frame.

## Folder Structer
### Calibration
data_object_calib/training/calib/ contains a textfile with the calibration matrices for each frame.

P0–P3: Projection matrices (3×4) projection matrices for the four cameras mounted on the KITTI vehicle:
The relevant matrix here is P2, as it meant for the left color camera, which is given in the dataset. The other 3 projection matrices can be ignored. It projects rectified camera coordinates → image pixel coordinates.

R0_rect: Rectification matrix (3×3) rotates points from the raw camera coordinate system to the rectified camera coordinate system.

Tr_velo_to_cam: LiDAR → Camera extrinsic transformation (3×4) converts 3D points from Velodyne LiDAR coordinates to the camera coordinate system.

Tr_imu_to_velo: IMU → LiDAR transformation (3×4) converts coordinates from the IMU frame (vehicle base) to the Velodyne frame. This is not relevant for this dataset.

To get a 3D point from the velo coordinate system into 2D image you need to first translate it into cam 3D coordinate system, rectify the point and project it into 2D image space (the order matters).

### Images
data_object_image_2/training/image_2/ contains the rgb image for each frame. It is the left color camera image of the KITTI dataset, which is normally used as the main camera. The other camera images were cut to make this dataset more simple.

### Ground Truth 3D bounding boxes
data_object_label_2/training/label_2/ contains the ground truth labels for each frame.
Each line is for one object, each line has the following values:
| Field | Description |
|-------|-------------|
| type |Object class, for example car|
| truncation| Fraction of object truncated out of image (0 = fully visible, 1 = fully outside)|
| occlusion| level of occlusion (0=fully visible, 1=partly occluded, 2=largely occluded, 3=unknown)|
|Alpha| Observation angle of object (in radians) in image plane, relative to camera optical axis in radians|
|x_min| Minimum x value of the of the 2D bounding box in image pixel coordinates|
|y_min| Minimum y value of the of the 2D bounding box in image pixel coordinates|
|x_max| Maximum x value of the of the 2D bounding box in image pixel coordinates|
|y_max| Maximum y value of the of the 2D bounding box in image pixel coordinates|
|height| Height of the object given in meters|
|width | Width of the object given in meters|
|length| Length of the object given in meters|
|x| X coordinate of the object center in the rectified camera coordinate system (same as for P2)|
|y| Y coordinate of the object center in the rectified camera coordinate system (same as for P2)|
|z| Z coordinate of the object center in the rectified camera coordinate system (same as for P2)|
|rotation y| Rotation around the Y-axis in the camera coordinate system, in radians|

### Velodyn Pointcloud
data_object_velodyne/training/velodyne/ contains the pointcloud of the velodyn LiDAR for each frame. The point cloud is saved as a bin (binary file). For each point the 3D coordinates and the intensity are given (meaning 4 values per point). You can load the file using np.fromfile("", dtype=np.float32). The coordinates are given in meters, the intensity as a value between 0 and 255. The raw .bin file just holds unstructured values, so probably want to to reshape the the 1D array to a an array of shape (N,4) using .reshape(-1,4). Consult the numpy documentation, for more details