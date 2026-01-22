import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def report_sampling_and_replay_buffer(*, replay_buffer):
    episodes_in_buffer = replay_buffer.get_num_episodes()
    ts_in_buffer = replay_buffer.get_num_timesteps()
    replayed_steps = replay_buffer.get_sampled_timesteps()
    added_steps = replay_buffer.get_added_timesteps()
    return {'BUFFER_capacity': replay_buffer.capacity, 'BUFFER_size_num_episodes': episodes_in_buffer, 'BUFFER_size_timesteps': ts_in_buffer, 'BUFFER_replayed_steps': replayed_steps, 'BUFFER_added_steps': added_steps}