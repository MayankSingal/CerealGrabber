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
    
    
    
    ######################################### Motion Planning ##########################################################
    descriptions, obs = task.reset()


    ## Motion 1
    ## (1) Sense mustard_grasp_point location. (2) Move gripper to a point 0.1m over mustard_grasp_point, while making it remain vertical.
    
    #(1)
    next_position, next_orientation = robot_controller.get_pose_from_simulation("mustard_grasp_point")
    
    #(2)
    next_position[2] += 0.1
    next_orientation = gripper_vertical_orientation
    robot_controller.move(next_position, next_orientation)
    
    ## Motion 2
    ## (1) Move downwards. (2) Close gripper
    
    #(1)
    next_position[2] -= 0.1
    next_orientation = gripper_vertical_orientation
    robot_controller.move(next_position, next_orientation)
    
    #(2)
    robot_controller.actuate_gripper(0)

    
    
    

    
    