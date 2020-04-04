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

    def get_poses(self, relative_to_gripper=False):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}
        
        for obj in objs:
            name = obj.get_name()
            if(relative_to_gripper == True):
                pose = obj.get_pose(relative_to=self._env._robot.gripper)
            else:
                pose = obj.get_pose()

            # pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            # gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            # perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            # pose[:3] += pos
            # pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses
    
    def get_bbox(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_bboxes = {}
        
        for obj in objs:
            name = obj.get_name()
            bbox = obj.get_bbox


if __name__ == "__main__":
    
    ### Set Action Mode
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
    
    ### Initialize Environment with Action Mode and desired observations
    env = Environment(action_mode, '', ObservationConfig(), False)
    
    ### Load task into the environment
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    
    ### Create Agent: TODO
    agent = RandomAgent()
    
    while True:
        ### Reset the task
        descriptions, obs = task.reset()
        
        ### Create Perception/Sensing Module
        obj_pose_sensor = NoisyObjectPoseSensor(env)
        
        ### Get target pose
        poses = obj_pose_sensor.get_poses()
        # pprint.pprint(poses)
        # brak
        target_pose = poses['soup_grasp_point']
        
        print("Going Down")
        ### Go Down
        req_position = env._robot.arm.get_tip().get_position()
        req_orientation = env._robot.arm.get_tip().get_orientation()
        req_position[2] -= 0.45
        path = env._robot.arm.get_linear_path(req_position, req_orientation) 
        path = path._path_points.reshape(-1, path._num_joints)
        
        for i in range(len(path)):
            
            action = list(path[i]) + [0.5]
            obs, reward, terminate = task.step(action)
        
        ### Use the in-built path planner to plan a path to the target object without collisions
        path = None
        target_object = None
        objs = env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        for obj in objs:
            if(obj.get_name() == "soup_grasp_point"):
                req_position = obj.get_position()
                req_orientation = obj.get_orientation()*-1
                req_position[2] += 0.1
                path = env._robot.arm.get_path(req_position, req_orientation, trials=10000) 
                target_object = obj
                break    
        path = path._path_points.reshape(-1, path._num_joints)
        
        
        ### Simulate
        for i in range(len(path)):
            
            action = list(path[i]) + [0.5]
            obs, reward, terminate = task.step(action)
            
        ### Go Down
        req_position[2] -= 0.1
        path = env._robot.arm.get_path(req_position, req_orientation, ignore_collisions=False, trials=10000) 
        path = path._path_points.reshape(-1, path._num_joints)
        
        for i in range(len(path)):
            
            action = list(path[i]) + [1]
            obs, reward, terminate = task.step(action)
            
            
            
        ### Grip the object
        for i in range(50):
            action = list(path[-1]) + [0]
            obs, reward, terminate = task.step(action)    
            
        ### Lift the object
        raised_position = target_object.get_position()
        raised_position[2] += 0.1
        raised_orientation = target_object.get_orientation()*-1
        path = env._robot.arm.get_path(raised_position, raised_orientation, ignore_collisions=True, trials=10000) 
        path = path._path_points.reshape(-1, path._num_joints)
        
        ### Simulate
        for i in range(len(path)):
            
            action = list(path[i]) + [0]
            obs, reward, terminate = task.step(action)
            
        ### Grip the object
        for i in range(50):
            action = list(path[-1]) + [1]
            obs, reward, terminate = task.step(action)    
        
        
    
        
    
        
        
    
        
        
    
     
    
    
    
    def old_knowledge():
        pass
        # poses = obj_pose_sensor.get_poses()
    
        # target_pose = poses['pick_and_lift_target']
        # robot_tip_pose = env._scene._robot.arm.get_tip().get_pose()
        
        # displacement_vector = target_pose[:3] - robot_tip_pose[:3]
        # disp_planned_delta = displacement_vector*0.005
        
        # full_planned_delta = list(disp_planned_delta) + [0,0,0,1,0.5]
        # # print(full_planned_delta)
            
        
        
        # desired_pos = robot_tip_pose[:3]
        
        # ct = 0
        # print(descriptions)
        # while True:
            
        #     all_poses = obj_pose_sensor.get_poses()
        #     robot_gripper_pose = env._scene._robot.arm.get_tip().get_pose()
        #     target_object_pose = all_poses['pick_and_lift_target']
        #     target_object_pose[2] += 0.2 
            
        #     disp_vector = target_object_pose[:3] - robot_gripper_pose[:3]
        #     disp_planned_delta = disp_vector*0.05
        #     action = list(disp_planned_delta) + [0,0,0,1] + [0.5]

        #     if(np.sqrt(disp_vector[0]**2 + disp_vector[1]**2 + disp_vector[2]**2) < 0.05):
        #         break        

        #     print(action)
        #     obs, reward, terminate = task.step(action)

        #     if terminate:
        #         break
            
        # print("Orienting EE")
        
        # env._action_mode = ArmActionMode.ABS_EE_POSE
        
        # while True:
            
        #     robot_gripper_pose = env._scene._robot.arm.get_tip().get_pose()
        #     target_gripper_pose = np.array(list(robot_gripper_pose))
        #     target_gripper_pose[3] = 0
            
        #     # print(robot_gripper_pose)

        #     planned_delta = list(target_gripper_pose) + [0.5]

        #     # planned_delta.append(0.1)
        #     #planned_delta[:3] = [0,0,0]
        #     obs, reward, terminate = task.step(planned_delta)
            
            
        
            
        # while True:
            
        #     all_poses = obj_pose_sensor.get_poses()
        #     robot_gripper_pose = env._scene._robot.arm.get_tip().get_pose()
        #     target_object_pose = all_poses['pick_and_lift_target']

        #     disp_vector = target_object_pose[:3] - robot_gripper_pose[:3]
        #     if(disp_vector[0]**2 + disp_vector[1]**2 + disp_vector[2]**2 < 0.001):
        #         break

        #     action = [0,0,-0.005] + [0,0,0,1] + [0.1]
        #     obs, reward, terminate = task.step(action)
            
        #     if(terminate):
        #         break
            
            

        # while True:
        #     action = [0,0,0] + [0,0,0,1] + [1]
        #     obs, reward, terminate = task.step(action)
        



    env.shutdown()
