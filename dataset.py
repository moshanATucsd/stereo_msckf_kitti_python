import numpy as np
import cv2
import os
import time
import pandas as pd
from collections import defaultdict, namedtuple
from threading import Thread

import utils 

class GroundTruthReader(object):
    def __init__(self, path, starttime=-float('inf')):
        self.df = pd.read_pickle(path)
        self.starttime = starttime
        self.field = namedtuple('gt_msg', ['timestamp', 'p', 'q', 'v'])

    def parse(self, i):
        """
        """
        timestamp = self.df.loc[i, "timestamp"]
        T_gt = self.df.loc[i, "T_i_vk"]
        p = T_gt[:3, 3]
        q = utils.to_quaternion(T_gt[:3, :3])
        v = self.df.loc[i, "v_vk_i_vk"]
        return self.field(timestamp, p, q, v)

    def set_starttime(self, starttime):
        self.starttime = starttime

    def __iter__(self):
        for i in range(len(self.df)):
            data = self.parse(i)
            if data.timestamp < self.starttime:
                continue
            yield data

    def get_poses(self):
        return np.array(list(self.df.loc[:, "T_i_vk"].values))

    def get_velocities(self):
        return np.array(list(self.df.loc[:, "v_vk_i_vk"]))

    def get_timestamps(self):
        return np.array(list(self.df.loc[:, "timestamp"]))

class IMUDataReader(object):
    def __init__(self, path):
        self.df = pd.read_pickle(path)
        self.starttime = -float('inf')
        self.field = namedtuple('imu_msg', 
            ['timestamp', 'angular_velocity', 'linear_acceleration'])

    def parse(self, i, j):
        """
        """
        timestamp = self.df.loc[i, "imu_timestamps"][j]
        wm = self.df.loc[i, "gyro_measurements"][j]
        am = self.df.loc[i, "accel_measurements"][j]
        # print("imu timestamps {}".format(timestamp))
        return self.field(timestamp, wm, am)

    def __iter__(self):
        for i in range(len(self.df)):
            for j in range(len(self.df.loc[i, "imu_timestamps"])):
                data = self.parse(i, j)
                if data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        first_start_time = self.df.loc[0, "imu_timestamps"][0]
        return first_start_time

    def set_starttime(self, starttime):
        self.starttime = starttime



class ImageReader(object):
    def __init__(self, img_path, data_path):
        self.ids = self.list_imgs(img_path)
        self.df = pd.read_pickle(data_path)

        timestamps = []
        for i in range(len(self.df)):
            timestamps.append(self.df.loc[i, "timestamp"])

        self.timestamps = timestamps
        self.starttime = self.start_time()

        self.cache = dict()
        self.idx = 0

        self.field = namedtuple('img_msg', ['timestamp', 'image'])

        self.ahead = 10   # 10 images ahead of current index
        self.wait = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        return cv2.imread(path, -1)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.wait:
                # print("insufficient time for imread")
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if self.timestamps[i] < self.starttime:
                    continue
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        if not self.thread_started:
            self.thread_started = True
            self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
            # print("image read cache")
        else:   
            img = self.read(self.ids[idx])
            # print("image read")
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            if timestamp < self.starttime:
                print("image continue")
                continue
            # print("image iter {}".format(i))
            yield self.field(timestamp, self[i])

    def start_time(self):
        first_start_time = self.timestamps[0]
        return first_start_time

    def set_starttime(self, starttime):
        self.starttime = starttime

    def list_imgs(self, dir):
        xs = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        xs = sorted(xs, key=lambda x:float(x[:-4]))
        return [os.path.join(dir, _) for _ in xs]


class Stereo(object):
    def __init__(self, cam0, cam1):
        assert len(cam0) == len(cam1)
        self.cam0 = cam0
        self.cam1 = cam1
        self.timestamps = cam0.timestamps

        self.field = namedtuple('stereo_msg', 
            ['timestamp', 'cam0_image', 'cam1_image', 'cam0_msg', 'cam1_msg'])

    def __iter__(self):
        for l, r in zip(self.cam0, self.cam1):
            assert abs(l.timestamp - r.timestamp) < 0.01, 'unsynced stereo pair'
            yield self.field(l.timestamp, l.image, r.image, l, r)

    def __len__(self):
        return len(self.cam0)

    def start_time(self):
        return self.cam0.starttime

    def set_starttime(self, starttime):
        self.starttime = starttime
        self.cam0.set_starttime(starttime)
        self.cam1.set_starttime(starttime)
        
    

class kittiDataset(object):   # Stereo + IMU
    '''
    path example: '/media/erl/disk2/kitti/2011_09_30/2011_09_30_drive_0027_extract'
    '''
    def __init__(self, img_path, data_path):
        self.groundtruth = GroundTruthReader(os.path.join(data_path, 'data.pickle'))
        self.imu = IMUDataReader(os.path.join(data_path, 'data.pickle'))

        self.cam0 = ImageReader(os.path.join(img_path, 'image_02', 'data'), os.path.join(data_path, 'data.pickle'))
        self.cam1 = ImageReader(os.path.join(img_path, 'image_03', 'data'), os.path.join(data_path, 'data.pickle'))

        self.stereo = Stereo(self.cam0, self.cam1)
        self.timestamps = self.cam0.timestamps

        self.starttime = max(self.imu.start_time(), self.stereo.start_time())
        self.set_starttime(0)

    def set_starttime(self, offset):
        self.groundtruth.set_starttime(self.starttime + offset)
        self.imu.set_starttime(self.starttime + offset)
        self.cam0.set_starttime(self.starttime + offset)
        self.cam1.set_starttime(self.starttime + offset)
        self.stereo.set_starttime(self.starttime + offset)


# simulate the online environment
class DataPublisher(object):
    def __init__(self, dataset, out_queue, duration=float('inf'), ratio=1.): 
        self.dataset = dataset
        self.dataset_starttime = dataset.starttime
        self.out_queue = out_queue
        self.duration = duration
        self.ratio = ratio
        self.starttime = None
        self.started = False
        self.stopped = False

        self.publish_thread = Thread(target=self.publish)
        
    def start(self, starttime):
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()

    def stop(self):
        self.stopped = True
        if self.started:
            self.publish_thread.join()
        self.out_queue.put(None)

    def publish(self):
        dataset = iter(self.dataset)
        while not self.stopped:
            try:
                data = next(dataset)
            except StopIteration:
                self.out_queue.put(None)
                return

            interval = data.timestamp - self.dataset_starttime
            if interval < 0:
                continue
            while (time.time() - self.starttime) * self.ratio < interval + 1e-3:
                time.sleep(1e-3)   # assumption: data frequency < 1000hz
                if self.stopped:
                    return

            if interval <= self.duration + 1e-3:
                self.out_queue.put(data)
            else:
                print("exception")
                self.out_queue.put(None)
                return

def print_msg(in_queue, source):
    while True:
        x = in_queue.get()
        if x is None:
            return
        print(x.timestamp, source)

if __name__ == '__main__':
    from queue import Queue

    img_path = '/media/erl/disk2/kitti/2011_09_30/2011_09_30_drive_0027_extract'
    data_path = '/home/erl/Workspace/deep_ekf_vio/data/K07'
    dataset = kittiDataset(img_path, data_path)
    dataset.set_starttime(offset=0)

    imu_queue = Queue()
    img_queue = Queue()
    # gt_queue = Queue()

    duration = 1

    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration)
    # gt_publisher = DataPublisher(
    #     dataset.groundtruth, gt_queue, duration)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    # gt_publisher.start(now)

    t2 = Thread(target=print_msg, args=(imu_queue, 'imu'))
    # t3 = Thread(target=print_msg, args=(gt_queue, 'groundtruth'))

    t2.start()
    # t3.start()

    timestamps = []
    while True:
        x = img_queue.get()
        if x is None:
            print("no image")
            break
        print(x.timestamp, 'image')
        # cv2.imshow('left', np.hstack([x.cam0_image, x.cam1_image]))
        # cv2.waitKey(1)
        timestamps.append(x.timestamp)

    imu_publisher.stop()
    img_publisher.stop()
    # gt_publisher.stop()

    t2.join()
    # t3.join()

    # print("elapsed time: {} s".format(time.time() - now))
    # print("dataset time interval: {} -> {}, {} s".format(timestamps[-1], timestamps[0], timestamps[-1]-timestamps[0]))
    # print('Please check if IMU and image are synced')