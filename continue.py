from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import gym
import tensorflow as tf  # Deep Learning library
from tensorflow import keras

env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n


def build_model(height, width, channels, actions):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                   )
    return dqn


model = build_model(height, width, channels, actions)
model.summary()
dqn = build_agent(model, actions)
dqn.compile(tf.keras.optimizers.Adam(lr=1e-4))
dqn.load_weights('spaceys.h5')
dqn.fit(env, nb_steps=20000, visualize=False, verbose=1)
dqn.save_weights('spaceys20.h5', overwrite=True)