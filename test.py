import gym

env_to_wrap = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env_to_wrap, "test", force=True)
frame = env.reset()
is_done = False
while not is_done:
  action = env.action_space.sample()
  _, _, is_done, _ = env.step(action)
env.close()
env_to_wrap.close()