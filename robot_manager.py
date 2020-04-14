import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from scipy.spatial.transform import Rotation as R

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutGroceriesInCupboard
from pyrep.const import ConfigurationPathAlgorithms as Algos
import pprint
import time

import copy

from utils import RandomAgent, NoisyObjectPoseSensor, VisionObjectPoseSensor, RobotController


if __name__ == "__main__":
    
    ### Set Action Mode
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
    
    ### Initialize Environment with Action Mode and desired observations
    env = Environment(action_mode, '', ObservationConfig(), False)
    
    ### Load task into the environment
    task = env.get_task(PutGroceriesInCupboard)
    
    ### Create Agent: TODO
    agent = RandomAgent()

    ### Object Pose Sensor
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    
    ### Robot Controller Object
    robot_controller = RobotController(env, task)
    
    ### Useful variables
    gripper_vertical_orientation = np.array([3.13767052e+00, 1.88300957e-03, 9.35417891e-01])
    reset_position = env._robot.arm.get_tip().get_position()
    reset_orientation = env._robot.arm.get_tip().get_orientation()
    
    mustard_orientation = [ 7.07037210e-01,  7.07173109e-01, -6.37740828e-04, -2.06269184e-03]
    mustard_position = [ 0.31035879, -0.12106754,  0.90185165]

    
    
    
    ######################################### Motion Planning ##########################################################
    descriptions, obs = task.reset()
    


    ## Motion 1
    ## (1) Sense mustard_grasp_point location. (2) Move gripper to a point 0.1m over mustard_grasp_point, while making it remain vertical.
    
    #(1)
    next_position, next_orientation, mustard_obj = robot_controller.get_pose_and_object_from_simulation("mustard_grasp_point")
    
    #(2)
    next_position[2] += 0.1
    next_orientation = gripper_vertical_orientation
    robot_controller.move(next_position, next_orientation)
        
    ## Motion 2
    ## (1) Move downwards. (2) Close gripper
    
    #(1)
    robot_controller.translate(z=-0.1)
    
    #(2)
    robot_controller.actuate_gripper(0)
    

    ## Motion 3
    # (1) Move upwards. (2) Move near mustard reset position
    
    # (1)
    robot_controller.translate(z=0.25, gripper_state=0, ignore_collisions=True)
    
    # (2)
    req_position = copy.copy(mustard_position)
    req_position[2] += 0.3
    req_position[0] -= 0.3
    robot_controller.move(req_position, gripper_vertical_orientation, gripper_state=0)
    
    
    ## Motion 4
    # (1) Rotate grasped object to vertical orientation.
    
    required_rotation = None   
    mustard_quat = R.from_quat(list(mustard_obj.get_quaternion()))
    desired_mustard_quat = R.from_quat(list(mustard_orientation))
    required_rotation = mustard_quat*desired_mustard_quat.inv()
    
    end_effector_orintation = R.from_euler('xyz', env._robot.arm.get_tip().get_orientation())
    new_orientation = end_effector_orintation * required_rotation
    new_orientation = list(new_orientation.as_euler('xyz'))
    
    robot_controller.rotate_to(new_orientation, gripper_state=0, ignore_collisions=True)
    
  
    ## Motion 5
    # (1) Move over mustard reset position, (2) Double down on object orientation
    
    # (1)
    req_position = copy.copy(mustard_position)
    req_position[2] += 0.3
    req_orientation = env._robot.arm.get_tip().get_orientation()
    robot_controller.move(req_position, req_orientation, gripper_state=0)
    
    # (2) 
    required_rotation = None   
    mustard_quat = R.from_quat(list(mustard_obj.get_quaternion()))
    desired_mustard_quat = R.from_quat(list(mustard_orientation))
    required_rotation = mustard_quat*desired_mustard_quat.inv()
    
    end_effector_orintation = R.from_euler('xyz', env._robot.arm.get_tip().get_orientation())
    new_orientation = end_effector_orintation * required_rotation
    new_orientation = list(new_orientation.as_euler('xyz'))
    
    robot_controller.rotate_to(new_orientation, gripper_state=0, ignore_collisions=True)
    
    
    ## Motion 6
    # (1) Move Down. (2) Drop Object
    
    # (1)
    robot_controller.translate(z=-0.3, gripper_state=0, ignore_collisions=True)
    
    # (2)
    robot_controller.actuate_gripper(1)
    
    


    
    
    
  
    


    
    
    

    
    