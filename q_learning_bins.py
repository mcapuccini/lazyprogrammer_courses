# %% Imports and defs
import numpy as np
import gym
import matplotlib.pyplot as plt

CART_POSITION_BINS = np.linspace(-2.4, 2.4, 9)
CART_VELOCITY_BINS = np.linspace(-2, 2, 9) 
POLE_ANGLE_BINS = np.linspace(-0.4, 0.4, 9)
POLE_VELOCITY_BINS = np.linspace(-3.5, 3.5, 9)

def obs2state(obs):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    features = [
        np.digitize(x=[cart_pos], bins=CART_POSITION_BINS)[0],
        np.digitize(x=[cart_vel], bins=CART_VELOCITY_BINS)[0],
        np.digitize(x=[pole_angle], bins=POLE_ANGLE_BINS)[0],
        np.digitize(x=[pole_vel], bins=POLE_VELOCITY_BINS)[0]
    ]
    return int("".join(map(lambda feature: str(int(feature)), features)))

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def play_episode(env, Q, eps, gamma=0.9, alpha=1e-2):
    done = False
    obs = env.reset()
    state = obs2state(obs)
    tot_reward = 0
    iters = 0
    while not done:
        action = np.argmax(Q[state])
        if np.random.random() < eps:
            action = env.action_space.sample()
        obs_prime, rew, done, _ = env.step(action)
        state_prime = obs2state(obs_prime)
        if done and iters < 199: # if game over before step 200 it's bad
            rew = -300
        Q[state, action] += alpha * (rew + gamma * np.max(Q[state_prime]) - Q[state, action])
        state = state_prime
        iters += 1
        tot_reward += rew
    return Q, tot_reward, iters

# %%
# Init
env = gym.make('CartPole-v0')
N = 10000
tot_rewards = np.empty(N)
num_states = 10**env.observation_space.shape[0]
num_actions = env.action_space.n
Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
# Main loop
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    Q, tot_reward, iters = play_episode(env, Q, eps)
    tot_rewards[n] = tot_reward
    if n % 100 == 0:
        print("episode:", n, "total reward:", tot_reward, "eps:", eps)
# Plot
plot_running_avg(tot_rewards)