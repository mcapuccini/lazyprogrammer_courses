import gym

env_to_wrap = gym.make("LunarLander-v2")
env = gym.wrappers.Monitor(env_to_wrap, "/host-home/Desktop/test", force=True)
frame = env_to_wrap.reset()
is_done = False
while not is_done:
  action = env_to_wrap.action_space.sample()
  _, _, is_done, _ = env.step(action)
env.close()
env_to_wrap.close()