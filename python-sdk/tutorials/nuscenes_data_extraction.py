from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt

def main():
    VERSION = 'v1.0-mini'
    DATA_DIR = '/media/jim/Hard Disk/nuscenes_data/sets/nuscenes'
    sensor = 'CAM_FRONT'
    render = True
    nusc = NuScenes(version=VERSION, dataroot=DATA_DIR, verbose=True)
    for scene in nusc.scene:
        print(f"processing scene {scene['name']}, ")
        print(f"{scene['description']}")
        sample_tokens_list = []
        current_sample = nusc.get('sample', scene['first_sample_token'])
        sample_tokens_list.append(current_sample['token'])
        while not current_sample['next'] == "":
            print(f"current sample token: {current_sample['token']}")
            cam_front_data = nusc.get('sample_data', current_sample['data'][sensor])
            calib_sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
            ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            if render:
                nusc.render_sample_data(cam_front_data['token'], with_anns=False)
                # plt.pause(1)
                # plt.close()
            filename = cam_front_data['filename']
            sensor_T = calib_sensor['translation']
            sensor_R = calib_sensor['rotation']
            camera_intrinsic = calib_sensor['camera_intrinsic']
            timestamp = ego_pose['timestamp']
            ego_R = ego_pose['rotation']
            ego_T = ego_pose['translation']
            print(filename)
            print(cam_front_data)
            print(ego_pose)
            # print(f"{timestamp}, {filename}, {camera_intrinsic}, {ego_R}")
            current_sample = nusc.get('sample', current_sample['next'])
            sample_tokens_list.append(current_sample['token'])


if __name__ == "__main__":
    main()