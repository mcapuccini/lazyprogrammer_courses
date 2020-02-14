import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad, SGD

N = 1000

class Policy:
    def __init__(self, in_size, out_size, hidden_sizes, lr=1e-1):
        in_l = Input(in_size)
        hl = in_l
        for size in hidden_sizes:
            hl = Dense(size, activation='tanh')(hl)
        out_l = Dense(out_size, activation='softmax', use_bias=False)(hl)
        self.out_size = out_size
        self.model = Model(in_l, out_l)
        self.model.build(in_size)
        self.optimizer = tf.keras.optimizers.Adagrad(1e-1)
        self.__train_on_batch_tf = tf.function(self.__train_on_batch,
            input_signature=[tf.TensorSpec(shape=(None, in_size), dtype=tf.float32),
                tf.TensorSpec(shape=None, dtype=tf.int32),
                tf.TensorSpec(shape=None, dtype=tf.float32)])
    
    def __train_on_batch(self, states, actions, advantages):
        def loss():
            pi = self.model(states)
            a_one_hot = tf.one_hot(actions, self.out_size)
            log_pi = tf.math.log(tf.reduce_sum(pi * a_one_hot, 1))
            return -tf.reduce_sum(log_pi * advantages)
        self.optimizer.minimize(loss, self.model.trainable_variables)

    def partial_fit(self, states, actions, advantages):
        states_2d = np.atleast_2d(states)
        advantages_1d = np.atleast_1d(advantages)
        actions_1d = np.atleast_1d(actions)
        return self.__train_on_batch_tf(states_2d, actions_1d, advantages_1d)

    @tf.function
    def __predict_tf(self, state):
        return self.model(state)

    def predict(self, state):
        state_2d = np.atleast_2d(state)
        return self.__predict_tf(state_2d).numpy()[0]

    def sample_action(self, state):
        p = self.predict(state)
        return np.random.choice(len(p), p=p)

class Value:
    def __init__(self, in_size, hidden_sizes, lr=1e-4):
        in_l = Input(in_size)
        hl = in_l
        for size in hidden_sizes:
            hl = Dense(size, activation='tanh')(hl)
        out_l = Dense(1)(hl)
        self.model = Model(in_l, out_l)
        self.model.build(in_size)
        self.optimizer = tf.keras.optimizers.SGD(1e-4)
        self.__train_on_batch_tf = tf.function(self.__train_on_batch,
            input_signature=[tf.TensorSpec(shape=(None, in_size), dtype=tf.float32),
                tf.TensorSpec(shape=None, dtype=tf.float32)])
    
    def __train_on_batch(self, states, returns):
        def loss():
            y_hat = tf.reshape(self.model(states), [-1])
            return tf.reduce_sum(tf.math.square(returns - y_hat))
        self.optimizer.minimize(loss, self.model.trainable_variables)

    def partial_fit(self, states, returns):
        states_2d = np.atleast_2d(states)
        returns_1d = np.atleast_1d(returns)
        return self.__train_on_batch_tf(states_2d, returns_1d)

    @tf.function
    def __predict_tf(self, state):
        return self.model(state)

    def predict(self, state):
        state_2d = np.atleast_2d(state)
        return self.__predict_tf(state_2d).numpy()[0]

def play_one_mc(env, policy, value, gamma=0.99):
    # Init
    total_rew=0
    done=False
    states = []
    actions = []
    rewards = []
    returns = []
    advantages = []
    obs = env.reset()
    r=0

    # While not game over
    while not done:
        # Sample action and store current (s,a,r) in memory
        a = policy.sample_action(obs)
        states.append(obs)
        actions.append(a)
        rewards.append(r)
        # Step
        obs_prime, r, done, _ = env.step(a)
        total_rew += r
        # If the pole fell
        if done:
            r = -200
        # Update obs
        obs = obs_prime
    
    # Save the final (s,a,r)
    action = policy.sample_action(obs)
    states.append(obs)
    actions.append(a)
    rewards.append(r)

    # Compute returns and advantages
    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - value.predict(s))
        G = r + gamma * G
    returns.reverse()
    advantages.reverse()

    # Update the models
    policy.partial_fit(states, actions, advantages)
    value.partial_fit(states, returns)

    # Return tot rew
    return total_rew

def play_one_ac(env, policy, value, gamma=0.99):
    # Init
    total_rew=0
    done=False
    obs = env.reset()

    # While not game over
    while not done:
        # Sample action
        a = policy.sample_action(obs)
        # Step
        obs_prime, r, done, _ = env.step(a)
        total_rew += r
        # Update the models
        V_prime = value.predict(obs_prime)
        G = r + gamma * V_prime
        advantage = G - value.predict(obs)
        policy.partial_fit(obs, a, advantage)
        value.partial_fit(obs, G)
        # Update obs
        obs = obs_prime
    
    # Return rew
    return total_rew

# Init
env = gym.make('CartPole-v0')
in_size = env.observation_space.shape[0]
out_size = env.action_space.n
policy = Policy(in_size, out_size, [])
value = Value(in_size, [10])
total_rewards = np.empty(N)
losses = np.empty(N)

# Main loop
for n in range(N):
    total_rewards[n] = play_one_mc(env, policy, value)
    if((n+1) % 10 == 0):
        print("episode:", n+1, "tot reward:", total_rewards[n], "avg reward (last 100):", total_rewards[max(0, n-100):(n+1)].mean())

# Plot rew
plt.plot(total_rewards)
running_avg = np.empty(len(total_rewards))
for t in range(len(total_rewards)):
    running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
plt.plot(running_avg)
plt.savefig("plot/r2_pg_rew.png")

# Save video for 1 episode
wrp = gym.wrappers.Monitor(env, "video/r2_pg", force=True)
play_one_ac(wrp, policy, value)
env.close()
wrp.close()