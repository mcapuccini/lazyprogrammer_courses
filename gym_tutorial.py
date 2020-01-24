import gym
import io
import base64
from IPython.display import HTML

env_to_wrap = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env_to_wrap, "video/gym_tutorial", force=True)
frame = env.reset()
is_done = False
while not is_done:
  action = env.action_space.sample()
  observation, reward, is_done, _ = env.step(action)
env.close()
env_to_wrap.close()