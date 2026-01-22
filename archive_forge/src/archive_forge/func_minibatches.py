import logging
import numpy as np
import random
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
@DeveloperAPI
def minibatches(samples: SampleBatch, sgd_minibatch_size: int, shuffle: bool=True):
    """Return a generator yielding minibatches from a sample batch.

    Args:
        samples: SampleBatch to split up.
        sgd_minibatch_size: Size of minibatches to return.
        shuffle: Whether to shuffle the order of the generated minibatches.
            Note that in case of a non-recurrent policy, the incoming batch
            is globally shuffled first regardless of this setting, before
            the minibatches are generated from it!

    Yields:
        SampleBatch: Each of size `sgd_minibatch_size`.
    """
    if not sgd_minibatch_size:
        yield samples
        return
    if isinstance(samples, MultiAgentBatch):
        raise NotImplementedError('Minibatching not implemented for multi-agent in simple mode')
    if 'state_in_0' not in samples and 'state_out_0' not in samples:
        samples.shuffle()
    all_slices = samples._get_slice_indices(sgd_minibatch_size)
    data_slices, state_slices = all_slices
    if len(state_slices) == 0:
        if shuffle:
            random.shuffle(data_slices)
        for i, j in data_slices:
            yield samples[i:j]
    else:
        all_slices = list(zip(data_slices, state_slices))
        if shuffle:
            random.shuffle(all_slices)
        for (i, j), (si, sj) in all_slices:
            yield samples.slice(i, j, si, sj)