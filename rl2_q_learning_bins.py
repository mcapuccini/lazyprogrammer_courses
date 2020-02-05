import numpy as np
import gym
import matplotlib.pyplot as plt

N = 10000
CART_POSITION_BINS = np.linspace(-2.4, 2.4, 9)
CART_VELOCITY_BINS = np.linspace(-2, 2, 9) 
POLE_ANGLE_BINS = np.linspace(-0.4, 0.4, 9)
POLE_VELOCITY_BINS = np.linspace(-3.5, 3.5, 9)

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def obs2state(obs):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    return build_state([
        to_bin(cart_pos, CART_POSITION_BINS),
        to_bin(cart_vel, CART_VELOCITY_BINS),
        to_bin(pole_angle, POLE_ANGLE_BINS),
        to_bin(pole_vel, POLE_VELOCITY_BINS),
    ])

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def eps_greedy(env, Q_state, eps):
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return np.argmax(Q_state)

def play_episode(env, Q, eps, gamma=0.9, alpha=1e-2):
    done = False
    obs = env.reset()
    state = obs2state(obs)
    tot_reward = 0
    while not done:
        # Step
        action = eps_greedy(env, Q[state], eps)
        obs_prime, rew, done, _ = env.step(action)
        # Update tot rew
        tot_reward += rew
        # Update Q
        if done and tot_reward < 199: # if game over before step 200 it's bad
            rew = -300
        state_prime = obs2state(obs_prime)
        Q[state, action] += alpha * (rew + gamma * np.max(Q[state_prime]) - Q[state, action])
        # Update state
        state = state_prime
    return Q, tot_reward

# Init
env = gym.make('CartPole-v0')
tot_rewards = np.empty(N)
num_states = 10**env.observation_space.shape[0]
num_actions = env.action_space.n
Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

# Main loop
for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    Q, tot_reward = play_episode(env, Q, eps)
    tot_rewards[n] = tot_reward
    if n % 100 == 0:
        print("episode:", n, "tot rew run avg:", tot_rewards[max(0, n-100):(n+1)].mean(), "eps:", eps)
        
# Plot
plot_running_avg(tot_rewards)