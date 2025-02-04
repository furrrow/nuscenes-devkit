from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import h5py
from tqdm import tqdm

"""
nuscences data extraction for both images and velocities
nuscenes devkit repo:
https://github.com/nutonomy/nuscenes-devkit/tree/master?tab=readme-ov-file#nuscenes

canbus documentation:
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md
"""

def append_to_dataset(scene_group, dataset_name, data):
    dataset = scene_group[dataset_name]
    dataset.resize((dataset.shape[0] + 1), axis=0)
    dataset[-1] = data

def save_h5(VERSION, generator, save_path):
    save_file_name = f"nuscenes_{VERSION}_combined.h5"
    with h5py.File(os.path.join(save_path, save_file_name), "w") as h5file:

        last_scene = None
        scene_group = None

        for img_paths, velocity, rotation_rate, acceleration, position, orientation, scene in generator:

            # If a new scene starts, create a new scene group
            if last_scene != scene:
                scene_group = h5file.require_group(f"scene_{scene}")
                last_scene = scene

                # Create resizable datasets
                scene_group.create_dataset("image_paths", shape=(0, 6), maxshape=(None, 6), dtype=h5py.string_dtype())
                scene_group.create_dataset("velocity", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)
                scene_group.create_dataset("rotation_rate", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)
                scene_group.create_dataset("acceleration", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)
                scene_group.create_dataset("position", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)
                scene_group.create_dataset("orientation", shape=(0, 4), maxshape=(None, 4), dtype=np.float32)

            # Save image paths as strings instead of actual images
            img_paths = [p.encode("utf-8") for p in img_paths]  # Convert to bytes
            append_to_dataset(scene_group, "image_paths", img_paths)
            append_to_dataset(scene_group, "velocity", velocity)
            append_to_dataset(scene_group, "rotation_rate", rotation_rate)
            append_to_dataset(scene_group, "acceleration", acceleration)
            append_to_dataset(scene_group, "position", position)
            append_to_dataset(scene_group, "orientation", orientation)

    print(f"{save_file_name} creation complete!")

def nuscenes_data_generator(VERSION, DATA_DIR, camera_sensors, verbose=False):
    """
    generator function for the nuscenes dataset
    Args:
        :param VERSION: nuscenes data version, eg: 'v1.0-mini'; 'v1.0-trainval'
        :param DATA_DIR: data root directory of the nuscenes dataset
        :param camera_sensors: list of sensors in nuscenes, currently only support list of cameras
        :param verbose: print/show images

    Returns:
        img_paths,
        velocity,
        rotation_rate,
        acceleration,
        position,
        orientation,
        scene_number

    """
    nusc = NuScenes(version=VERSION, dataroot=DATA_DIR, verbose=verbose)
    nusc_can = NuScenesCanBus(dataroot=DATA_DIR)
    blacklist = nusc_can.can_blacklist
    for i_scene, scene in tqdm(enumerate(nusc.scene), total=len(nusc.scene), disable=verbose):

        scene_number = int(scene['name'][-4:])
        if scene_number in blacklist:
            print("scene found in blacklist, skipping...")
            continue

        scene_name = scene['name']

        if verbose:
            print("==================================")
            print(f"processing {scene_name}, {i_scene + 1} out of {len(nusc.scene)} scenes")
            print(f"description: {scene['description']}")

        camera_tokens = {
            cam: nusc.get('sample_data', nusc.get('sample', scene['first_sample_token'])['data'][cam])['token']
            for cam in camera_sensors
        }

        # Load CAN bus pose data
        pose_data = nusc_can.get_messages(scene['name'], 'pose')
        img_paths = []

        while all(camera_tokens.values()):

            if verbose:
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            for i, cam in enumerate(camera_sensors):
                cam_token = camera_tokens[cam]

                cam_data = nusc.get('sample_data', cam_token)

                if verbose:
                    print(f"{cam} Timestamp: {cam_data['timestamp']}")

                cam_path = cam_data['filename']

                # Load image
                img_paths.append(cam_path)

                if verbose:
                    # Display image
                    full_cam_path = os.path.join(DATA_DIR, cam_path)
                    img = Image.open(full_cam_path)
                    row, col = divmod(i, 3)
                    axs[row, col].imshow(img)
                    axs[row, col].set_title(f"{cam}")
                    axs[row, col].axis('off')

                # If CAM_FRONT, get the closest pose data
                if cam == 'CAM_FRONT' and pose_data:
                    closest_pose = min(pose_data, key=lambda x: abs(x['utime'] - cam_data['timestamp']))

                    # Extract pose information
                    acceleration = np.array(closest_pose['accel'])
                    orientation = np.array(closest_pose['orientation'])
                    position = np.array(closest_pose['pos'])
                    rotation_rate = np.array(closest_pose['rotation_rate'])
                    velocity = np.array(closest_pose['vel'])

                    if verbose:
                        print("\n--- Closest Pose Data ---")
                        print(f"Timestamp: {closest_pose['utime']}")
                        print(f"Acceleration (m/s²): {acceleration}")
                        print(f"Orientation (quaternion): {orientation}")
                        print(f"Position (x, y, z in meters): {position}")
                        print(f"Rotation Rate (rad/s): {rotation_rate}")
                        print(f"Velocity (m/s): {velocity}")

                        print(cam_data)
                if cam_data['next']:
                    camera_tokens[cam] = cam_data['next']
                else:
                    camera_tokens[cam] = None

            if verbose:
                print("\n")
                plt.show()
            img_paths = np.array(img_paths)
            yield img_paths, velocity, rotation_rate, acceleration, position, orientation, scene_number
            img_paths = []

def main():
    VERSION = 'v1.0-mini'
    # VERSION = 'v1.0-trainval'
    DATA_DIR = '/media/jim/Hard Disk/nuscenes_data/sets/nuscenes'
    save_path = "/home/jim/Documents/Projects/nuscenes-devkit"
    sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    generator = nuscenes_data_generator(VERSION, DATA_DIR, sensors, verbose=False)
    save_h5(VERSION, generator, save_path)

if __name__ == "__main__":
    main()