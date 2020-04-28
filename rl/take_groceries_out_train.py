from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutGroceriesInCupboard
import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion
from scipy.spatial.transform import Rotation as R
from pyrep.const import ConfigurationPathAlgorithms as Algos
import pprint
import time

from utils import RandomAgent, NoisyObjectPoseSensor, VisionObjectPoseSensor, RobotController,rotation_transform
from DQN_take_object_out_class import*


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [0.0]  # Always close
        return np.concatenate([arm, gripper], axis=-1)



if __name__ == "__main__":

    #Create RL agent 
    RLagent = rl_take_out()


    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
    env = Environment(
        action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(PutGroceriesInCupboard)
    robot_controller = RobotController(env, task)
    


    RLagent = rl_take_out()
#     RLagent.agent.load(directory="model")

    training_steps = 120000
    episode_length = 100
    save_freq = 10000
    obs = None
    
    #Getting the reset point position
    reset_position, reset_orientation, reset_obj = robot_controller.get_pose_and_object_from_simulation("target_waypoint")
    
    for i in range(training_steps):
        if i % episode_length == 0:
            
            descriptions, obs = task.reset()

    #         #(1) Go to waypoint3 outside of cupboard
    #         next_position, next_orientation, _ = robot_controller.get_pose_and_object_from_simulation("waypoint3")
    #         try:
    #             robot_controller.move(next_position, next_orientation, 1)
    #         except:
    #             continue 
    #         waypoint3_orient = next_orientation
    #         #(2) Reach to the spam and close gripper
    #         tar_init_pos, tar_verticle_orientation, target_obj = robot_controller.get_pose_and_object_from_simulation("spam_grasp_point")
    #         try:
    #             path = robot_controller.move(tar_init_pos, waypoint3_orient,1 ,ignore_collisions=True)
    #         except:
    #             continue
    #         robot_controller.actuate_gripper(0)
    #         robot_controller.move_with_path(np.flip(path,0),0)
        
    #     #Getting grasp success state 
    #     grasp_success = robot_controller.get_grasp_state()
        
    #     # Getting the updated target position and orientation 
    #     tar_position = list(target_obj.get_position())
    #     tar_orientation = list(target_obj.get_orientation())
    #     #Getting the arm pose and
    #     arm_pose = robot_controller.get_arm_tip_pose()  
        
    #     state = tar_position + tar_orientation
        
    #     action = RLagent.get_action(state)
        
        action = [0,0,0,0,0,0,0,0]
        

        obs, _, _ = task.step(action)
#         time.sleep(0.5)
        
        
#         reward, terminal = RLagent.get_reward(tar_position, reset_position, tar_orientation, tar_verticle_orientation, tar_init_pos,grasp_success)
# #         print(reward)
#         print("random action", np.around(action, decimals=1))

#         RLagent.agent.observe(terminal=terminal, reward=reward)
        
        
        if i % save_freq == save_freq - 1: RLagent.agent.save(directory="model")






















    # training_steps = 1200
    # episode_length = 40
    # obs = None
    # for i in range(training_steps):
    #     if i % episode_length == 0:
    #         print('Reset Episode')
    #         descriptions, obs = task.reset()

    #         #(1) Go to waypoint3 outside of cupboard
    #         next_position, next_orientation, _ = robot_controller.get_pose_and_object_from_simulation("waypoint3")
    #         robot_controller.move(next_position, next_orientation, 1)
    #         waypoint3_orient = next_orientation
    #         #(2) Reach to the spam and close gripper
    #         next_position, next_orientation, _ = robot_controller.get_pose_and_object_from_simulation("spam_grasp_point")
    #         path = robot_controller.move(next_position, waypoint3_orient,1 ,ignore_collisions=True)
    #         robot_controller.actuate_gripper(0)

    #         print(descriptions)
    #     action = agent.act(obs)
    #     robot_controller.actuate_gripper(0)
    #     obs, reward, terminate = task.step(action)
    #     print("Reward", reward)
    #     print("Terminate", terminate)

    print('Done')
    env.shutdown()
