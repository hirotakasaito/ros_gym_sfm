import sys, os
import time
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
        self.lidar_pub = rospy.Publisher("ros_gym_sfm/scan", LaserScan, queue_size=1)
        self.odom_pub = rospy.Publisher("ros_gym_sfm/odom", PoseStamped, queue_size=10)
        self.grand_truth_pose = rospy.Publisher("ros_gym_sfm/grand_truth_pose", PoseStamped, queue_size=10)

        #subscriber
        self.cmd_vel = rospy.Subscriber("ros_gym_sfm/cmd_vel", Twist, self.cmd_vel_callback)

    def cmd_vel_callback(self, cmd_vel):
        self.linear_x = cmd_vel.linear.x
        self.angular_z = cmd_vel.angular.z

    def process(self):
        rate = rospy.Rate(self.HZ)
        observation = self.env.reset()
        action = np.random.rand(2)
        while not rospy.is_shutdown():
            scan, _, _ ,_ =self.env.step(action)
            self.env.render()

        self.env.close()

if __name__ == '__main__':
    ros_gym_sfm = RosGymSfm()

    try:
        ros_gym_sfm.process()
    except rospy.ROSInterruptException:
        pass
