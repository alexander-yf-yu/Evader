import tensorflow as tf
from train import eval_env, eval_policy, tf_agent, actor_net, optimizer

# suppress warning about CPU usage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WHICH_TO_RESTORE = 1

checkpoint = tf.train.Checkpoint(actor_net=actor_net, optimizer=optimizer)

checkpoint_directory = "tmp/training_checkpoints/full_train"

manager = tf.train.CheckpointManager(checkpoint,
                                     directory=checkpoint_directory,
                                     checkpoint_name='save',
                                     max_to_keep=20)

restore_path = manager.checkpoints[WHICH_TO_RESTORE - 1]

checkpoint.restore(restore_path)

tf_agent.initialize()

while True:
    time_step = eval_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        action_step = eval_policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        episode_return += time_step.reward

    print(episode_return.numpy()[0])





