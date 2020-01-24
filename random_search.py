# %% Imports and defs
import gym
import numpy as np
import matplotlib.pyplot as plt

def play_episode(env, w):
    done = False
    t = 0
    s = env.reset()
    while not done and t < 10000:
        action = 1 if s.dot(w) > 0 else 0
        s, _, done, _ = env.step(action)
        t += 1
    return t

def play_T_episodes(env, T, w):
    episode_lengths = np.empty(T)
    for i in range(T):
        episode_lengths[i] = play_episode(env, w)
    avg_length = episode_lengths.mean()
    print("avg length:", avg_length)
    return avg_length

def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for _ in range(100):
        new_params = np.random.random(4) * 2 - 1
        avg_length = play_T_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)
        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params

# %% Play!
env = gym.make('CartPole-v0')
episode_lengths, params = random_search(env)

# %% Plot
plt.plot(episode_lengths)
plt.show()

# %% Play last episode set
play_T_episodes(env, 100, params)

# %% Play one more episode and save video
wrp = gym.wrappers.Monitor(env, "video/random_search.py", force=True)
play_T_episodes(wrp, 1, params)
wrp.close()