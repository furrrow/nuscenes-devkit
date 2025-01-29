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

def closest_timestep_interpolate(reference_array, interpolate_array):
    match_array = []
    interp_idx = 0
    for ref_idx, entry in enumerate(reference_array):
        i_timestamp, i_value = entry
        i_timestamp = int(i_timestamp)
        if i_timestamp < interpolate_array[interp_idx][0]:
            print(f"skipping timestep {i_timestamp}")
            reference_array = reference_array[ref_idx+1:]
            continue
        time_diff = np.abs(interpolate_array[:, 0] - i_timestamp)
        closest_match = np.argmin(time_diff)
        match_array.append(interpolate_array[closest_match])
        # print(f"found match ref {ref_idx} interp {closest_match}")
    return reference_array, np.array(match_array)

def main():
    # VERSION = 'v1.0-mini'
    VERSION = 'v1.0-trainval'
    DATA_DIR = '/media/jim/Hard Disk/nuscenes_data/sets/nuscenes'
    save_path = "/home/jim/Documents/Projects/nuscenes-devkit"
    dataset_name = "nuscenes"
    sensor = 'CAM_FRONT'
    camera_data = {}
    nusc = NuScenes(version=VERSION, dataroot=DATA_DIR, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=DATA_DIR)
    with h5py.File(os.path.join(save_path, f"{dataset_name}_combined.h5"), "w") as h5file:
        for scene in nusc.scene:
            scene_name = scene['name']
            print("==================================")
            print(f"processing scene {scene_name}, ")
            print(f"{scene['description']}")
            scene_group = h5file.require_group(f"scene_{scene_name}")
            sample_tokens_list = []
            timestamps = []
            files_list = []
            current_sample = nusc.get('sample', scene['first_sample_token'])
            cam_front_data = nusc.get('sample_data', current_sample['data'][sensor])
            calib_sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
            camera_intrinsic = calib_sensor['camera_intrinsic']
            camera_data[scene_name] = camera_intrinsic
            # print(f"{cam_front_data['calibrated_sensor_token']}")
            while not current_sample['next'] == "":
                # print(f"current sample token: {current_sample['token']}")
                sample_tokens_list.append(current_sample['token'])
                cam_front_data = nusc.get('sample_data', current_sample['data'][sensor])
                assert(cam_front_data['calibrated_sensor_token'] == calib_sensor['token'])
                ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
                filename = cam_front_data['filename']
                timestamp = int(ego_pose['timestamp'])
                timestamps.append(timestamp)
                files_list.append(filename)
                # ego_R = ego_pose['rotation']
                # ego_T = ego_pose['translation']
                current_sample = nusc.get('sample', current_sample['next'])
            timestamps = np.array(timestamps)
            files_list = np.array(files_list)
            camera_files = np.stack((timestamps, files_list)).T

            # look at extracted image time freq
            analyze_series(timestamps)

            # get imu and velocities
            ms_imu = nusc_can.get_messages(scene_name, 'ms_imu') # 100hz
            ms_imu_a_r = np.array([(m['utime'], m['linear_accel'][0], m['linear_accel'][1], m['linear_accel'][2],
                                    m['rotation_rate'][0], m['rotation_rate'][1], m['rotation_rate'][2])
                                   for m in ms_imu])
            # steer angle feedback, Steering angle feedback in radians in range [-7.7, 6.3].
            # 0 indicates no steering, positive values indicate left turns, negative values right turns.
            angle = nusc_can.get_messages(scene_name, 'steeranglefeedback')
            angle = np.array([(m['utime'], m['value']) for m in angle])

            # Zoe Vehicle Info, 100hz
            zoe_veh = nusc_can.get_messages(scene_name, 'zoe_veh_info')
            wheel_speed = np.array([(m['utime'], m['FL_wheel_speed']) for m in zoe_veh])
            # Convert to m/s.
            radius = 0.305  # Known Zoe wheel radius in meters.
            circumference = 2 * np.pi * radius
            wheel_speed[:, 1] *= circumference / 60

            updated_files, updated_speed = closest_timestep_interpolate(reference_array=camera_files, interpolate_array=wheel_speed)
            updated_files, updated_imu = closest_timestep_interpolate(reference_array=updated_files, interpolate_array=ms_imu_a_r)
            updated_files, updated_angle = closest_timestep_interpolate(reference_array=updated_files, interpolate_array=angle)
            print(f"lengths of combined files")
            # print(len(updated_files), len(updated_angle), len(updated_speed), len(updated_imu))
            print(len(updated_files))

            x_vel = np.cos(updated_angle[:, 1]) * updated_speed[:, 1]
            y_vel = -np.sin(updated_angle[:, 1]) * updated_speed[:, 1]
            z_vel = np.zeros_like(updated_angle[:, 1])
            print(f"saving scene {scene_name}...")
            for i in range(len(updated_files)):
                img_path = os.path.join(DATA_DIR, updated_files[i][1])
                img = cv2.imread(img_path)
                a_omega = updated_imu[i][1:7] # ax, ay, az, wx, wy, yz
                v_a_omega = [x_vel[i], y_vel[i], z_vel[i]] + list(a_omega)
                # Create a new pair within the scene group
                pair_group = scene_group.create_group(f"pair_{i}")
                # Save the image as a dataset
                pair_group.create_dataset("image", data=img, dtype=np.uint8)

                # Save the IMU data as a dataset
                pair_group.create_dataset("imu", data=np.array(v_a_omega, dtype=float))
            # break
    # saving camera intrinsics data
    json_object = json.dumps(camera_data, indent=4)
    json_file_path = os.path.join(save_path, f"{dataset_name}_camera_data.json")
    with open(json_file_path, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()