import sys, os
import time
import math
import numpy as np
import random
import traceback
import argparse
import gym
import gym_sfm.envs.env as envs
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

parser = argparse.ArgumentParser()
parser.add_argument('--map', help='Specify map setting folder.', default='demo')
parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1800)
parser.add_argument('-mt', '--max_t', type=int, default=1800)
parser.add_argument('-mepi', '--max_episodes', type=int, default=5)
args = parser.parse_args()

# env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)

# for i_episode in range(args.max_episodes):
#     observation = env.reset()
#     # env.agent.pose = np.array([0.0, 0.0])
#     done = False
#     epidode_reward_sum = 0
#     # start = time.time()
#     for t in range(args.max_t):
#         action = np.array([1, 0], dtype=np.float64)
#         observation, reward, done, _ = env.step(action)
#         # print(observation)
#         env.render()
#         # print(done)
#         # if not done :
#         #     # env.close()
#         #     break
#     env.close()
#     # end = time.time()
#     # print(end - start)
# print('Finished all episode.')


class RosGymSfm:

    def __init__(self):
        rospy.init_node("ros_gym_sfm")

        self.env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)
        self.HZ = 40;

        #publisher
        self.scan_pub = rospy.Publisher("ros_gym_sfm/scan", LaserScan, queue_size=1)
        self.odom_pub = rospy.Publisher("ros_gym_sfm/odom", PoseStamped, queue_size=10)
        self.grand_truth_pose = rospy.Publisher("ros_gym_sfm/grand_truth_pose", PoseStamped, queue_size=10)

        #subscriber
        self.cmd_vel = rospy.Subscriber("ros_gym_sfm/cmd_vel", Twist, self.cmd_vel_callback)

        self.scan = LaserScan()
        self.scan.header.frame_id = "laser"
        self.scan.angle_min = -2.356194496
        self.scan.angle_max = 2.356194496
        self.scan.angle_increment = 0.00436
        self.scan.time_increment = 1.7361
        self.scan.scan_time = 0.02500
        self.scan.range_max = 60
        self.scan.range_min = 0.023

        self.pose = PoseStamped()

    def cmd_vel_callback(self, cmd_vel):
        self.linear_x = cmd_vel.linear.x
        self.angular_z = cmd_vel.angular.z

    def transform_pose(self, pose, base_pose):
        x = pose[0] - base_pose[0]
        y = pose[1] - base_pose[1]
        yaw = pose[2] -base_pose[2]
        trans_pose = np.array([
            x*np.cos(base_pose[2]) + y*np.sin(base_pose[2]),
            -x*np.sin(base_pose[2]) + y*np.cos(base_pose[2]),
            np.arctan2(np.sin(yaw), np.cos(yaw))
        ])
        return trans_pose

    def process(self):
        rate = rospy.Rate(self.HZ)
        observation = self.env.reset()
        action = np.random.rand(2)
        base_pose_flag = False

        while not rospy.is_shutdown():
            scan, _, _,pose,yaw ,_ =self.env.step(action)
            yaw = yaw % (2*math.pi)
            pose = np.append(pose,yaw)

            if base_pose_flag == False:
                base_pose = pose
                base_pose_flag = True

            trans_pose = self.transform_pose(pose, base_pose)
            print(trans_pose)
            self.scan.ranges = scan
            self.scan_pub.publish(self.scan)
            self.env.render()

        self.env.close()

if __name__ == '__main__':
    ros_gym_sfm = RosGymSfm()

    try:
        ros_gym_sfm.process()

    except rospy.ROSInterruptException:
        pass
