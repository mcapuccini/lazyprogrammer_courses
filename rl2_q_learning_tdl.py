import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

N = 300

class TDLRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.e = np.zeros(D)

    def update_e(self, grad, gamma, lmbd=0.7):
        self.e = grad + self.e * gamma * lmbd

    def partial_fit(self, state, G, lr=1e-2):
        delta = G - self.predict(state)
        self.w += lr * delta * self.e

    def predict(self, state):
        X = np.array(state)
        return X.dot(self.w)

class ObsToState:
    def __init__(self, env, n_components=500):
        self.scaler = StandardScaler()
        self.fu = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
        ])
        obs_samples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler.fit(obs_samples)
        tmp_ft = self.fu.fit_transform(self.scaler.transform(obs_samples))
        self.dimensions = tmp_ft.shape[1]

    def transform(self, obs):
        scaled = self.scaler.transform([obs])
        transformed = self.fu.transform(scaled)
        return transformed[0]

def eps_greedy(env, Q_state, eps):
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return np.argmax(Q_state)

def play_episode(env, Q, obs2state, eps, gamma=0.9999):
    done = False
    obs = env.reset()
    state = obs2state.transform(obs)
    Q_state = np.stack([q.predict(state) for q in Q]).T
    tot_reward = 0
    while not done:
        # Step
        action = eps_greedy(env, Q_state, eps)
        obs_prime, rew, done, _ = env.step(action)
        # Update tot rew
        tot_reward += rew
        # Update Q
        state_prime = obs2state.transform(obs_prime)
        Q_state_prime = np.stack([q.predict(state_prime) for q in Q]).T
        G = rew + gamma * np.max(Q_state_prime)
        for i in range(env.action_space.n):
            if(i == action):
                Q[i].update_e(state, gamma)
            else:
                Q[i].update_e(0, gamma)
        Q[action].partial_fit(state, [G]) 
        # Update state
        state = state_prime
        Q_state = Q_state_prime
    return Q, tot_reward

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

# Init
env = gym.make('MountainCar-v0')
obs2state = ObsToState(env)
Q = []
for i in range(env.action_space.n):
    q = TDLRegressor(obs2state.dimensions)
    q.partial_fit(obs2state.transform(env.reset()), [0]) # opt. initial values
    Q.append(q)
tot_rewards = np.empty(N)

# Main loop
for n in range(N):
    eps = 0.1*(0.97**n)
    Q, tot_reward = play_episode(env, Q, obs2state, eps)
    tot_rewards[n] = tot_reward
    print("episode:", n, "tot rew:", tot_reward, "eps:", eps)

# Plot
plt.plot(tot_rewards)
plt.title("Total rewards")
plt.show()
plot_running_avg(tot_rewards)

# Save video for 1 episode
wrp = gym.wrappers.Monitor(env, "video/q_learning_rbf", force=True)
play_episode(wrp, Q, obs2state, 0)
env.close()
wrp.close()