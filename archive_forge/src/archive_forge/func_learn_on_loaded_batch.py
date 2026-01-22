import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@DeveloperAPI
def learn_on_loaded_batch(self, offset: int=0, buffer_index: int=0):
    """Runs a single step of SGD on an already loaded data in a buffer.

        Runs an SGD step over a slice of the pre-loaded batch, offset by
        the `offset` argument (useful for performing n minibatch SGD
        updates repeatedly on the same, already pre-loaded data).

        Updates the model weights based on the averaged per-device gradients.

        Args:
            offset: Offset into the preloaded data. Used for pre-loading
                a train-batch once to a device, then iterating over
                (subsampling through) this batch n times doing minibatch SGD.
            buffer_index: The index of the buffer (a MultiGPUTowerStack)
                to take the already pre-loaded data from. The number of buffers
                on each device depends on the value of the
                `num_multi_gpu_tower_stacks` config key.

        Returns:
            The outputs of extra_ops evaluated over the batch.
        """
    raise NotImplementedError