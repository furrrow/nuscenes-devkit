from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import h5py
import cv2

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

def main():
    VERSION = 'v1.0-mini'
    # VERSION = 'v1.0-trainval'
    DATA_DIR = '/media/jim/Hard Disk/nuscenes_data/sets/nuscenes'
    save_path = "/home/jim/Documents/Projects/nuscenes-devkit"
    dataset_name = "nuscenes"
    sensors = ['CAM_FRONT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    calibration_data = {}
    nusc = NuScenes(version=VERSION, dataroot=DATA_DIR, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=DATA_DIR)
    blacklist = nusc_can.can_blacklist
    with h5py.File(os.path.join(save_path, f"{dataset_name}_combined.h5"), "w") as h5file:
        for i_scene, scene in enumerate(nusc.scene):
            scene_name = scene['name']
            print("==================================")
            print(f"processing {scene_name}, {i_scene} out of {len(nusc.scene)} scenes")
            print(f"{scene['description']}")

            scene_number = int(scene['name'][-4:])
            if scene_number in blacklist:
                print("scene found in blacklist, skipping...")
                continue

            scene_group = h5file.require_group(f"scene_{scene_name}")
            sample_tokens_list = []
            timestamps = []
            files_list = []
            camera_intrinsic_dict = {}
            current_sample = nusc.get('sample', scene['first_sample_token'])
            for sensor_name in sensors:
                sensor_data = nusc.get('sample_data', current_sample['data'][sensor_name])
                calib_sensor = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
                camera_intrinsic = calib_sensor['camera_intrinsic']
                camera_intrinsic_dict[sensor_name] = camera_intrinsic
            calibration_data[scene_name] = camera_intrinsic_dict
            # print(f"{cam_front_data['calibrated_sensor_token']}")
            while not current_sample['next'] == "":
                # print(f"current sample token: {current_sample['token']}")
                sample_tokens_list.append(current_sample['token'])
                sensor_files = []
                for sensor_name in sensors:
                    sensor_data = nusc.get('sample_data', current_sample['data'][sensor_name])
                    calib_sensor = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
                    assert(sensor_data['calibrated_sensor_token'] == calib_sensor['token'])
                    if sensor_name == 'CAM_FRONT':
                        ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
                    filename = sensor_data['filename']
                    sensor_files.append(filename)
                timestamp = int(ego_pose['timestamp'])
                timestamps.append(timestamp)
                # ego_R = ego_pose['rotation']
                # ego_T = ego_pose['translation']
                current_sample = nusc.get('sample', current_sample['next'])
                files_list.append(sensor_files)
            timestamps = np.array(timestamps)
            files_list = np.array(files_list)

            # look at extracted image time freq
            analyze_series(timestamps)

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
            print(f"saving scene {scene_name}...")
            for i in range(len(files_list)):
                img_paths = [os.path.join(DATA_DIR, img_loc) for img_loc in files_list[i]]
                images = [cv2.imread(img_path) for img_path in img_paths]
                pos = updated_pose[i][1]
                accel = updated_pose[i][3]
                rot_rate = updated_pose[i][4]
                output_list = list(pos) + [x_vel[i], y_vel[i], z_vel[i]] + list(accel) + list(rot_rate)
                # Create a new pair within the scene group
                pair_group = scene_group.create_group(f"pair_{i}")
                # Save the image as a dataset
                pair_group.create_dataset("images", data=images, dtype=np.uint8)

                # Save the IMU data as a dataset
                pair_group.create_dataset("pose", data=np.array(output_list, dtype=float))
            # break
    # saving camera intrinsics data
    json_object = json.dumps(calibration_data, indent=4)
    json_file_path = os.path.join(save_path, f"{dataset_name}_camera_data.json")
    with open(json_file_path, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()