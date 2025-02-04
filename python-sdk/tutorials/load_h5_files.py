import json
import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

class CustomNuscenesDataset(Dataset):
    def __init__(self, data_file, camera_json, data_root, transform=None, target_transform=None):
        self.mode = "files"
        self.data_file = data_file
        self.camera_json = camera_json
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.camera_meta = json.load(open(camera_json))
        self.cam_scene_names = list(self.camera_meta.keys())
        self.cam_sensor_names = list(self.camera_meta[self.cam_scene_names[0]].keys())
        with h5py.File(data_file, 'r') as file:
            self.file_scene_list = list(file.keys())
    def __len__(self):
        return len(self.file_scene_list)

    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as file:
            scene_name = self.file_scene_list[idx]
            assert scene_name[-10:] == self.cam_scene_names[idx][0]
            pair_keys = list(file[scene_name].keys())
            first_pair = file[scene_name][pair_keys[0]]
            images = first_pair['image']
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def read_nuscenes_h5(data_file, camera_json, data_root):
    camera_meta = json.load(open(camera_json))
    cam_scene_names = list(camera_meta.keys())
    cam_sensor_names = list(camera_meta[cam_scene_names[0]].keys())
    print(camera_meta[cam_scene_names[0]])
    with h5py.File(data_file, 'r') as file:
        for i, scene_name in enumerate(list(file.keys())):
            assert scene_name[-10:] == cam_scene_names[0]
            pair_keys = list(file[scene_name].keys())
            first_pair = file[scene_name][pair_keys[0]]
            images = first_pair['image']
            assert len(images) == len(cam_sensor_names)
            pos = first_pair['imu'][0:3]
            vel = first_pair['imu'][3:6]
            accel = first_pair['imu'][6:9]
            rot_rate = first_pair['imu'][9:12]
            camera_matrices = camera_meta[cam_scene_names[0]]
            print(pos, vel, accel, rot_rate)
            print(camera_matrices)
            if type(images[i]) is np.ndarray:
                print(len(images), images[i].shape)
            else:
                img_path = str(images[i])
                print(os.path.join(data_root, img_path))
            break

def torch_load_nuscenes_h5(data_file, camera_json):
    camera_meta = json.load(open(camera_json))
    cam_scene_names = list(camera_meta.keys())
    cam_sensor_names = list(camera_meta[cam_scene_names[0]].keys())
    print(camera_meta[cam_scene_names[0]])
    with h5py.File(data_file, 'r') as file:
        for i, scene_name in enumerate(list(file.keys())):
            assert scene_name[-10:] == cam_scene_names[0]
            pair_keys = list(file[scene_name].keys())
            first_pair = file[scene_name][pair_keys[0]]
            images = first_pair['image']
            assert len(images) == len(cam_sensor_names)
            pos = first_pair['imu'][0:3]
            vel = first_pair['imu'][3:6]
            accel = first_pair['imu'][6:9]
            rot_rate = first_pair['imu'][9:12]
            camera_matrices = camera_meta[cam_scene_names[0]]
            print(len(images), images[0].shape)
            print(pos, vel, accel, rot_rate)
            print(camera_matrices)
            break


if __name__ == "__main__":
    data_root = '/media/jim/Hard Disk/nuscenes_data/sets/nuscenes'
    # data_file = "/home/jim/Documents/Projects/nuscenes-devkit/nuscenes_mini_images_1_combined.h5"
    data_file = "/home/jim/Documents/Projects/nuscenes-devkit/nuscenes_mini_files_all_combined.h5"
    # camera_json = "/home/jim/Documents/Projects/nuscenes-devkit/nuscenes_mini_images_1_camera_data.json"
    camera_json = "/home/jim/Documents/Projects/nuscenes-devkit/nuscenes_mini_files_all_camera_data.json"
    read_nuscenes_h5(data_file, camera_json, data_root)