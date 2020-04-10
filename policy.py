from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import EvaderEnv as Env
import tensorflow as tf
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# suppress warning about GPU usage
from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyper parameters
num_iterations = 1000  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}

fc_layer_params = (200, 100)

learning_rate = 0.001  # @param {type:"number"}
num_eval_episodes = 5  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}

tf.compat.v1.enable_v2_behavior()

t_env = Env()
e_env = Env()

Env.graphics = False

train_env = tf_py_environment.TFPyEnvironment(t_env)
eval_env = tf_py_environment.TFPyEnvironment(e_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter
)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0

    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity
)


def collect_episode(environment, policy, num_episodes):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step,
                                          next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# Reset the train step
tf_agent.train_step_counter.assign(0)

greedy = []
collect = []

print("Pre-training demo:")
compute_avg_return(eval_env, tf_agent.policy, 5)

print("Beginning Training...")

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    experience = replay_buffer.gather_all()
    train_loss = tf_agent.train(experience)
    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    print("Training episode: {0}".format(step))

    if step % eval_interval == 0:
        Env.graphics = True
        print("___Policy Evaluation___")
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        print("Evaluating Greedy Policy...")
        avg_greedy = compute_avg_return(eval_env, tf_agent.policy, 5)
        print('step = {0}: Greedy Avg Return = {1}'.format(step, avg_greedy))
        greedy.append(avg_greedy)
        print("Evaluating Collection Policy...")
        avg_collect = compute_avg_return(train_env, tf_agent.collect_policy, 5)
        print('step = {0}: Collection Avg Return = {1}'.format(step, avg_collect))
        collect.append(avg_greedy)
        print("___Resuming Training___")
        Env.graphics = False

        # Breakout of training if reward > 1500
        if avg_greedy > 1500:
            break


tf_agent.train_step_counter.assign(0)

# Data
print("\nTotal training episodes: {0}".format(step))

print("\nPolicy Rewards: ")
for i in range(len(greedy)):
    episode = (i + 1) * eval_interval
    print("Greedy at episode {0}: reward = {1}".format(episode, greedy[i]))
    print("Collection at episode {0}: reward = {1}".format(episode, collect[i]))



