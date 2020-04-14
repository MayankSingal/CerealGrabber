import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from pyrep.const import ConfigurationPathAlgorithms as Algos
import pickle

import pprint
import time

def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
        
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz


class RandomAgent:

    def act(self, obs):
        delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0]
        delta_quat = [0, 0, 0, 1] # xyzw
        gripper_pos = [np.random.rand() > 0.5]
        return delta_pos + delta_quat + gripper_pos


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

    def get_poses(self, relative_to_gripper=False, add_noise=False):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}
        
        for obj in objs:
            name = obj.get_name()
            if(relative_to_gripper == True):
                pose = obj.get_pose(relative_to=self._env._robot.gripper)
            else:
                pose = obj.get_pose()

            if add_noise:
                pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
                gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
                perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

                pose[:3] += pos
                pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses
    
    def get_bbox(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_bboxes = {}
        
        for obj in objs:
            name = obj.get_name()
            bbox = obj.get_bbox


if __name__ == "__main__":
    
    dataset = {}
    
    ### Set Action Mode
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
    
    ### Initialize Environment with Action Mode and desired observations
    env = Environment(action_mode, '', ObservationConfig(), False)
    
    ### Load task into the environment
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    
    ### Create Agent: TODO
    agent = RandomAgent()
    
    for i in range(1000):
        ### Reset the task
        descriptions, obs = task.reset()
        
        ### Create Perception/Sensing Module
        obj_pose_sensor = NoisyObjectPoseSensor(env)
        
        ### Get target pose
        poses = obj_pose_sensor.get_poses()
       
        crackers_grasp_point_pose = poses['crackers_grasp_point']
        soup_grasp_point_pose = poses['soup_grasp_point']
        tuna_grasp_point_pose = poses['tuna_grasp_point']
        pose_data = np.stack([crackers_grasp_point_pose, soup_grasp_point_pose, tuna_grasp_point_pose])
        
        wrist_rgb = obs.wrist_rgb
        ls_rgb = obs.left_shoulder_rgb
        rs_rgb = obs.right_shoulder_rgb
        
        image_data = np.stack([wrist_rgb, ls_rgb, rs_rgb])
        
        dataset[i] = (image_data, pose_data)
        
        print(i, "data samples collected")
        
        if((i+1) % 10 == 0):
            with open("database2.pkl", 'wb') as pfile:
                pickle.dump(dataset, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        
            
        
        
        
