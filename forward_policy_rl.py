import gym
import rlbench.gym
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

env = gym.make('put_groceries_in_cupboard-state-v0', render_mode='human')

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)
model.save("yolo")