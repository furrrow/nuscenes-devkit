from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import h5py
import cv2
from tqdm import tqdm

"""
nuscences data extraction for both images and velocities
nuscenes devkit repo:
https://github.com/nutonomy/nuscenes-devkit/tree/master?tab=readme-ov-file#nuscenes

canbus documentation:
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md
"""

def analyze_series(timed_array):
    last_time = int(timed_array[-1])
    first_time = int(timed_array[0])
    print("duration of scene:")
    print((last_time - first_time) / (1e6))
    print("number of entries:")
    print(len(timed_array))
    # print("averaged time diff in miliseconds:")
    # print((last_time - first_time) / len(timed_array))
    print("total frequency:")  #  -> close enough to 100 hz!!
    print(len(timed_array) / (last_time - first_time) * (1e6))

def closest_timestep_interpolate(reference_times, interpolate_array):
    match_array = []
    for i_timestamp in reference_times:
        i_timestamp = int(i_timestamp)
        time_diff = np.abs(interpolate_array[:, 0] - i_timestamp)
        closest_match = np.argmin(time_diff)
        match_array.append(interpolate_array[closest_match])
    return np.array(match_array)

def save_h5(generator, h5_path, dataset_name="nuscenes"):
    with h5py.File(os.path.join(h5_path, f"{dataset_name}_combined.h5"), "w") as h5file:


        for images, imu, scene in tqdm(generator):

            # Create or get the scene group
            scene_group = h5file.require_group(f"scene_{scene}")

            # Create a new pair within the scene group
            pair_idx = len(scene_group)  # Count existing pairs to assign a unique name
            pair_group = scene_group.create_group(f"pair_{pair_idx}")

            # Save the image as a dataset
            pair_group.create_dataset("image", data=images, dtype=np.uint8)

            # Save the IMU data as a dataset
            pair_group.create_dataset("imu", data=np.array(imu, dtype=float))

    print("HDF5 file creation complete!")

def nuscenes_data_generator(VERSION, DATA_DIR, sensors, save_path, dataset_name="nuscenes"):
    calibration_data = {}
    nusc = NuScenes(version=VERSION, dataroot=DATA_DIR, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=DATA_DIR)
    blacklist = nusc_can.can_blacklist
    for i_scene, scene in enumerate(nusc.scene):
        scene_name = scene['name']
        print("==================================")
        print(f"processing {scene_name}, {i_scene} out of {len(nusc.scene)} scenes")
        print(f"{scene['description']}")

        scene_number = int(scene['name'][-4:])
        if scene_number in blacklist:
            print("scene found in blacklist, skipping...")
            continue

        timestamps = []
        files_list = []
        camera_intrinsic_dict = {}
        current_sample = nusc.get('sample', scene['first_sample_token'])
        for sensor_name in sensors:
            sensor_files = []
            sensor_data = nusc.get('sample_data', current_sample['data'][sensor_name])
            calib_sensor = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
            camera_intrinsic = calib_sensor['camera_intrinsic']
            camera_intrinsic_dict[sensor_name] = camera_intrinsic
            while not sensor_data['next'] == "":
                if sensor_name == 'CAM_FRONT':
                    timestamps.append(sensor_data['timestamp'])
                filename = sensor_data['filename']
                sensor_files.append(filename)
                sensor_data = nusc.get('sample_data', sensor_data['next'])
            files_list.append(sensor_files)
        calibration_data[scene_name] = camera_intrinsic_dict
        timestamps = np.array(timestamps)

        # look at extracted image time freq
        # analyze_series(timestamps)

        # get imu and velocities
        # ms_imu = nusc_can.get_messages(scene_name, 'ms_imu') # 100hz
        # ms_imu_a_r = np.array([(m['utime'], m['linear_accel'][0], m['linear_accel'][1], m['linear_accel'][2],
        #                         m['rotation_rate'][0], m['rotation_rate'][1], m['rotation_rate'][2])
        #                        for m in ms_imu])
        # steer angle feedback, 100hz, Steering angle feedback in radians in range [-7.7, 6.3].
        # 0 indicates no steering, positive values indicate left turns, negative values right turns.
        angle = nusc_can.get_messages(scene_name, 'steeranglefeedback')
        angle = np.array([(m['utime'], m['value']) for m in angle])

        # pose Info, 50hz
        pose_info = nusc_can.get_messages(scene_name, 'pose')
        # pose_speed = np.array([(m['utime'], m['vel']) for m in pose_info])
        pose_array = np.array([(m['utime'], m['pos'], np.linalg.norm(m['vel']), m['accel'], m['rotation_rate']) for m in pose_info])

        updated_angle = closest_timestep_interpolate(reference_times=timestamps, interpolate_array=angle)
        updated_pose = closest_timestep_interpolate(reference_times=timestamps, interpolate_array=pose_array)
        print(f"lengths of combined files")
        # print(len(updated_files), len(updated_angle), len(updated_speed), len(updated_imu))
        print(len(updated_pose))

        updated_speed = updated_pose[:, 2]
        x_vel = np.cos(updated_angle[:, 1]) * updated_speed
        y_vel = -np.sin(updated_angle[:, 1]) * updated_speed
        z_vel = np.zeros_like(updated_angle[:, 1])
        # print(f"saving scene {scene_name}...")
        for i in range(len(files_list)):
            img_paths = [os.path.join(DATA_DIR, img_loc) for img_loc in files_list[i]]
            images = [cv2.imread(img_path) for img_path in img_paths]
            pos = updated_pose[i][1]
            accel = updated_pose[i][3]
            rot_rate = updated_pose[i][4]
            output_list = list(pos) + [x_vel[i], y_vel[i], z_vel[i]] + list(accel) + list(rot_rate)
            yield images, output_list, scene_name

    # saving camera intrinsics data
    json_object = json.dumps(calibration_data, indent=4)
    print(f"saving calibration data in {dataset_name}_camera_data.json")
    json_file_path = os.path.join(save_path, f"{dataset_name}_camera_data.json")
    with open(json_file_path, "w") as outfile:
        outfile.write(json_object)

def main():
    VERSION = 'v1.0-mini'
    # VERSION = 'v1.0-trainval'
    DATA_DIR = '/media/jim/Hard Disk/nuscenes_data/sets/nuscenes'
    save_path = "/home/jim/Documents/Projects/nuscenes-devkit"
    sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    save_name = "nuscenes_test"
    generator = nuscenes_data_generator(VERSION, DATA_DIR, sensors, save_path, save_name)
    save_h5(generator, save_path, save_name)

if __name__ == "__main__":
    main()