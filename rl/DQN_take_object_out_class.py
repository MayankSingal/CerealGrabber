from tensorforce import Agent
import numpy as np
import math



class rl_take_out(object):
    def __init__(self, target = "spam_grasp_point"):
        super().__init__()
        self.target = target
        self.action_size = 7
        self.explore = 0.4
        self.len_episode = 40
        self.terminal_threshold = 0.5
        self.input_high = 1.0
        self.input_low = 0.0
        self.agent = self.create_RLagent()
        

    def create_RLagent(self):
        num_states = 6
        num_inputs = self.action_size
        input_high = self.input_high
        input_low  = self.input_low

        states_dict = {'type': 'float', 'shape': num_states}
        actions_dict = {'type': 'float', 'shape': num_inputs, 'min_value': input_low, 'max_value': input_high}


        agent = Agent.create(
                    agent='dqn',
                    states = states_dict,  # alternatively: states, actions, (max_episode_timesteps)
                    actions = actions_dict,
                    memory=20000,
                    max_episode_timesteps= self.len_episode,
                    exploration= self.explore
                    )
        return agent

    # def get_in_state():




    def get_action(self,in_state):
        gripper = [0.0]  # Always close
        action = self.agent.act(states=in_state)
        if self.explore > np.random.uniform():
            action = np.random.uniform(self.input_low,self.input_high, size=(self.action_size))
        
        action = action * math.pi- math.pi/2
        actions = list(action) + gripper
        
        return actions
    


    def get_reward(self ,target_pos, reset_pos , tar_orientation, tar_verticle_orientation,tar_init_pos,grasp_success):
        terminal = False
        pos_dist = np.linalg.norm(target_pos-reset_pos)
        orient_dist = np.linalg.norm(tar_orientation-tar_verticle_orientation)
        
        total_dist = pos_dist+orient_dist
        
        #making sure that the object got moved
        if np.linalg.norm(target_pos-tar_init_pos)>0.005 and grasp_success: 
            reward = (1/pos_dist)*100 + (1/orient_dist)*10
        else:
            reward = 0
 
        
        if total_dist < self.terminal_threshold:
            terminal = True



        return reward , terminal 
