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
@PublicAPI(stability='alpha')
def restore_connectors(self, state: PolicyState):
    """Restore agent and action connectors if configs available.

        Args:
            state: The new state to set this policy to. Can be
                obtained by calling `self.get_state()`.
        """
    from ray.rllib.connectors.util import restore_connectors_for_policy
    if not self.config.get('enable_connectors', False):
        return
    connector_configs = state.get('connector_configs', {})
    if 'agent' in connector_configs:
        self.agent_connectors = restore_connectors_for_policy(self, connector_configs['agent'])
        logger.debug('restoring agent connectors:')
        logger.debug(self.agent_connectors.__str__(indentation=4))
    if 'action' in connector_configs:
        self.action_connectors = restore_connectors_for_policy(self, connector_configs['action'])
        logger.debug('restoring action connectors:')
        logger.debug(self.action_connectors.__str__(indentation=4))