import tensorflow as tf
from policy import optimizer, actor_net, BREAKOUT_REWARD, compute_avg_return, \
    collect_episode, train_env, eval_env

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer

# suppress warning about CPU usage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyper parameters
num_iterations = 1000  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}
save_interval = 100

tf.compat.v1.enable_v2_behavior()

train_step_counter = tf.compat.v2.Variable(0)

pre_train_checkpoint = tf.train.Checkpoint(actor_net=actor_net,
                                           optimizer=optimizer)

checkpoint_directory = "tmp/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "pre_train")

manager = tf.train.CheckpointManager(pre_train_checkpoint,
                                     directory=checkpoint_prefix,
                                     checkpoint_name='pre_train',
                                     max_to_keep=20)


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

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity
)

# Reset the train step
tf_agent.train_step_counter.assign(0)

print("Evaluating base policy:")
pre_train_avg = compute_avg_return(eval_env, tf_agent.policy)
print("Base return: {0}\n".format(pre_train_avg))

manager.save()

greedy = []
collect = []

print("Beginning Training...")

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_env, tf_agent.collect_policy,
                    collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    experience = replay_buffer.gather_all()
    train_loss = tf_agent.train(experience)
    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    print("Training episode: {0}".format(step))

    if step % save_interval == 0:
        manager.save()

    if step % eval_interval == 0:
        print("\n___Policy Evaluation___")
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        print("Evaluating Greedy Policy...")
        avg_greedy = compute_avg_return(eval_env, tf_agent.policy)
        print('step = {0}: Greedy Avg Return = {1}'.format(step, avg_greedy))
        greedy.append(avg_greedy)
        print("Evaluating Collection Policy...")
        avg_collect = compute_avg_return(train_env, tf_agent.collect_policy)
        print(
            'step = {0}: Collection Avg Return = {1}'.format(step, avg_collect))
        collect.append(avg_greedy)
        print("___Resuming Training___\n")

        # Breakout of training if reward > BREAKOUT_REWARD
        if avg_greedy > BREAKOUT_REWARD:
            break

train_env.close()
eval_env.close()

# Data
print("\nTotal training episodes: {0}".format(step))

print("\nPolicy Rewards: ")
for i in range(len(greedy)):
    episode = (i + 1) * eval_interval
    print("Greedy at episode {0}: reward = {1}".format(episode, greedy[i]))
    print("Collection at episode {0}: reward = {1}".format(episode, collect[i]))
