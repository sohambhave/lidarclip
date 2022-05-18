from os.path import join
from typing import Dict, List, Tuple

import numpy as np

from torch.utils.data import Dataset

from once_devkit.once import ONCE


CAM_NAMES = ["cam0%d" % cam_num for cam_num in (1, 3, 5, 6, 7, 8, 9)]


class OnceImageLidarDataset(Dataset):
    def __init__(self, data_root: str, img_transform=None):
        super().__init__()
        self._data_root = join(data_root, "data")
        self._devkit = ONCE(data_root)
        self._frames = self._setup()
        self._img_transform = img_transform

    def _setup(self) -> List[Tuple[str, str, str, Dict]]:
        mega_sequence_dict = {
            **self._devkit.val_info,
            # **self._devkit.train_info,
            # ...
        }
        frames = []
        for sequence_id, seq_info in mega_sequence_dict.items():
            for frame_id, frame_info in seq_info.items():
                # frame value (not used) has 'pose', 'calib', 'annos'
                for cam_name in self._devkit.camera_names:
                    frames.append((sequence_id, frame_id, cam_name, frame_info))
        return frames

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        """Load image and point cloud.

        The point cloud undergoes the following:
        - transformed to camera
        - all points with negative z-coords (behind camera plane) are removed
        - coordinate system is converted to KITTI-style x-forward, y-left, z-up

        """
        sequence_id, frame_id, cam_name, frame_info = self._frames[index]
        image = self._devkit.load_image(sequence_id, frame_id, cam_name)
        if self._img_transform:
            image = self._img_transform(image)
        point_cloud = self._devkit.load_point_cloud(sequence_id, frame_id)
        calib = frame_info["calib"][cam_name]
        point_cloud = self._transform_lidar_to_cam(point_cloud, calib)
        # TODO: maybe crop cloud
        return image, point_cloud

    def _transform_lidar_to_cam(self, points_lidar, calibration):
        cam_2_lidar = calibration["cam_to_velo"]
        point_xyz = points_lidar[:, :3]
        points_homo = np.hstack(
            [
                points_lidar[:, :3],
                np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1)),
            ]
        )
        points_cam = np.dot(points_homo, np.linalg.inv(cam_2_lidar).T)
        mask = points_cam[:, 2] > 0
        points_cam = points_cam[mask]  # discard points behind camera
        # Convert from openc camera coordinates to KITTI style (x-forward, y-left, z-up)
        point_cam_with_reflectance = np.hstack(
            [
                points_cam[:, 2:3],  # z -> x
                -points_cam[:, 0:1],  # -x -> y
                -points_cam[:, 1:2],  # -y -> z
                points_lidar[mask][:, 3:],  # add original reflectance
            ]
        )
        return point_cam_with_reflectance


def build_loader(clip_preprocess):
    raise NotImplementedError


def demo_dataset():
    import matplotlib.pyplot as plt

    datadir = "/Users/s0000960/data/once"
    dataset = OnceImageLidarDataset(datadir)
    image, lidar = dataset[0]
    plt.imshow(image)
    plt.show()
    plt.figure(figsize=(10, 10), dpi=200)
    # for visualization convert to x-right, y-forward
    plt.scatter(-lidar[:, 1], lidar[:, 0], s=0.1, c=np.clip(lidar[:, 3], 0, 1), cmap="coolwarm")
    plt.axis("equal")
    plt.xlim(-10, 10)
    plt.ylim(0, 40)
    plt.show()


if __name__ == "__main__":
    demo_dataset()