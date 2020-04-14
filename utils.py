import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from pyrep.const import ConfigurationPathAlgorithms as Algos

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

    def get_poses(self, relative_to_wrist_cam=False):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}
        
        for obj in objs:
            name = obj.get_name()
            if(relative_to_wrist_cam == True):
                pose = obj.get_pose(relative_to=self._env._scene._cam_wrist)
            else:
                pose = obj.get_pose()

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
            bbox = obj.get_bounding_box()
            
            obj_bboxes[name] = bbox
        
        return obj_bboxes    
        
            
            
class VisionObjectPoseSensor:
    
    def __init__(self):
        
        self.model = net()
        self.model.load_state_dict(torch.load("best_model.pth"))
        
    def inferPose(self, image_wrist, image_ls, image_rs):
        
        image_wrist = torch.from_numpy(image_wrist).float().permute(2,0,1).unsqueeze(0)
        image_ls = torch.from_numpy(image_ls).float().permute(2,0,1).unsqueeze(0)
        image_rs = torch.from_numpy(image_rs).float().permute(2,0,1).unsqueeze(0)
        
        predictions = self.model(image_wrist, image_ls, image_rs).data.cpu().numpy()
        
        return predictions[:3], predictions[3:6], predictions[6:]
    
    
class RobotController:
    
    def __init__(self, env, task):
        self.env = env
        self.task = task
        
    def get_pose_and_object_from_simulation(self, name):
        
        target_obj = None
        objs = self.env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        for obj in objs:
            if(obj.get_name() == name):
                req_position = obj.get_position()
                req_orientation = obj.get_orientation()
                target_obj = obj
                break
            
        return req_position, req_orientation, target_obj
            
    def actuate_gripper(self, gripper_state=0):

        joint_positions = self.env._robot.arm.get_joint_positions()
        
        for i in range(25):
            action = list(joint_positions) + [gripper_state]
            obs, reward, terminate = self.task.step(action)
    
            
    
    def translate(self, x=0, y=0, z=0, gripper_state=1, ignore_collisions=False):
        
        if(x == 0 and y == 0 and z == 0):
            print("No position change requested in translate.")
            return
        
        tip_position = self.env._robot.arm.get_tip().get_position()
        tip_orientation = self.env._robot.arm.get_tip().get_orientation()
        
        requested_position = tip_position + np.array([x,y,z])
        path = self.env._robot.arm.get_path(requested_position, tip_orientation, ignore_collisions=ignore_collisions)
        path = path._path_points.reshape(-1, path._num_joints)
                
                
        gripper = gripper_state
        
        for i in range(len(path)):

            action = list(path[i]) + [gripper]
            obs, reward, terminate = self.task.step(action)
            
        return path
            
    
    def rotate_to(self, orientation=None, gripper_state=1, ignore_collisions=False):
        if(orientation is None):
            print("No rotation requested.")
            return
        print("Rotating")
        tip_position = self.env._robot.arm.get_tip().get_position()
        tip_orientation = self.env._robot.arm.get_tip().get_orientation()
        
        requested_orientation = orientation
       
        path = self.env._robot.arm.get_path(tip_position, requested_orientation, ignore_collisions=ignore_collisions, trials=1000)
        path = path._path_points.reshape(-1, path._num_joints)
       
        gripper = gripper_state
        for i in range(len(path)):

            action = list(path[i]) + [gripper]
            obs, reward, terminate = self.task.step(action)
            
        return path
            
       

        
        
            
                
        
    def move(self, position=None, orientation=None, gripper_state=1, ignore_collisions=False):
        
        
        if(ignore_collisions == True):
            print("Planning, not checking for collisions...")
        else:
            print("Planning, with collision checks")
        
        path = None
        
        if(position is None or orientation is None):
            print("Position or orientation not specified. Check inputs.")
            return
        elif(gripper_state is None):
            print("Moving. Gripper defaults to open (1)")
            path = self.env._robot.arm.get_path(position, orientation, ignore_collisions=ignore_collisions)
            path = path._path_points.reshape(-1, path._num_joints)
        else:
            print("Moving to desired position, orientation and gripper state")
            path = self.env._robot.arm.get_path(position, orientation, ignore_collisions=ignore_collisions)
            path = path._path_points.reshape(-1, path._num_joints)
            
        if(path is None):
            print("No path received..")
            return
        
        gripper = gripper_state
        
        ## Execute the planned path
        for i in range(len(path)):

            action = list(path[i]) + [gripper]
            obs, reward, terminate = self.task.step(action)

        return path
    
    