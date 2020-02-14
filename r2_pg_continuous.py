import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_probability.python.distributions import Normal

N = 50

class FeatureTransformer:
  def __init__(self, env, n_components=100):
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    featurizer = FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
    ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
  def transform(self, observations):
    scaled = self.scaler.transform(observations)
    return self.featurizer.transform(scaled)

class Policy:
    def __init__(self, ft, in_size, hidden_sizes, lr=1e-3, smooth=1e-5):
        # Save attributes
        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.ft = ft
        self.lr = lr
        self.smooth = smooth
        self.optimizer = AdamOptimizer(lr)
        # NN architecture
        in_l = Input(in_size)
        hl = in_l
        for size in hidden_sizes:
            hl = Dense(size, activation='tanh')(hl)
        # Mean out layer
        out_l_mean = Dense(1, activation='tanh', use_bias=False, kernel_initializer='zeros')(hl)
        self.mean_model = Model(in_l, out_l_mean)
        self.mean_model.build(in_size)
        # Std dev out layer
        out_l_std = Dense(1, activation='softplus', use_bias=False)(hl)
        self.std_model = Model(in_l, out_l_std)
        self.std_model.build(in_size)
        # Traininng operation
        self.__train_on_batch_tf = tf.function(self.__train_on_batch,
            input_signature=[tf.TensorSpec(shape=(None, in_size), dtype=tf.float32),
                tf.TensorSpec(shape=None, dtype=tf.float32),
                tf.TensorSpec(shape=None, dtype=tf.float32)])

    def __train_on_batch(self, states, actions, advantages):
        def loss():
            mean = tf.reshape(self.mean_model(states), [-1])
            std = tf.reshape(self.std_model(states) + self.smooth, [-1])
            norm = Normal(mean, std)
            log_probs = norm.log_prob(actions)
            return -tf.reduce_sum(advantages * log_probs + 0.1 * norm.entropy())
        self.optimizer.minimize(loss)

    def partial_fit(self, states, actions, advantages):
        states_2d = np.atleast_2d(states)
        features = self.ft.transform(states_2d)
        advantages_1d = np.atleast_1d(advantages)
        actions_1d = np.atleast_1d(actions)
        return self.__train_on_batch_tf(features, actions_1d, advantages_1d)

    @tf.function
    def __predict_tf(self, state):
        mean = tf.reshape(self.mean_model(state), [-1])
        std = tf.reshape(self.std_model(state) + self.smooth, [-1])
        norm = Normal(mean, std)
        return tf.clip_by_value(norm.sample(), -1, 1)

    def predict(self, state):
        state_2d = np.atleast_2d(state)
        features = self.ft.transform(state_2d)
        return self.__predict_tf(features).numpy()[0]

    def sample_action(self, state):
        return self.predict(state)

class Value:
    def __init__(self, ft, in_size, hidden_sizes, lr=1e-1):
        self.ft = ft
        in_l = Input(in_size)
        hl = in_l
        for size in hidden_sizes:
            hl = Dense(size, activation='tanh')(hl)
        out_l = Dense(1)(hl)
        self.model = Model(in_l, out_l)
        self.model.build(in_size)
        self.optimizer = Adam(lr)
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
        features = self.ft.transform(states_2d)
        return self.__train_on_batch_tf(features, returns_1d)

    @tf.function
    def __predict_tf(self, state):
        return self.model(state)

    def predict(self, state):
        state_2d = np.atleast_2d(state)
        features = self.ft.transform(state_2d)
        return self.__predict_tf(features).numpy()[0]

def play_one_ac(env, policy, value, gamma=0.95):
    # Init
    total_rew=0
    done=False
    obs = env.reset()

    # While not game over
    while not done:
        # Sample action
        a = policy.sample_action(obs)
        # Step
        obs_prime, r, done, _ = env.step([a])
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
env = gym.make('MountainCarContinuous-v0')
ft = FeatureTransformer(env)
policy = Policy(ft, ft.dimensions, [])
value = Value(ft, ft.dimensions, [])
total_rewards = np.empty(N)
losses = np.empty(N)

# Main loop
for n in range(N):
    total_rewards[n] = play_one_ac(env, policy, value)
    print("episode:", n+1, "tot reward:", total_rewards[n], "avg reward (last 100):", total_rewards[max(0, n-100):(n+1)].mean())

# Plot rew
plt.plot(total_rewards)
running_avg = np.empty(len(total_rewards))
for t in range(len(total_rewards)):
    running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
plt.plot(running_avg)
plt.savefig("plot/r2_pg_continuous.png")

# Save video for 1 episode
wrp = gym.wrappers.Monitor(env, "video/r2_pg_continuous", force=True)
play_one_ac(wrp, policy, value)
env.close()
wrp.close()
