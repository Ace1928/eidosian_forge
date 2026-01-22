from collections import defaultdict
import logging
import time
import tree  # pip install dm_tree
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv
from ray.rllib.env.external_env import ExternalEnvWrapper
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv, get_wrapper_by_cls
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollectorGroup
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch, get_original_space
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def create_episode(self, env_id: EnvID) -> EpisodeV2:
    """Creates a new EpisodeV2 instance and returns it.

        Calls `on_episode_created` callbacks, but does NOT reset the respective
        sub-environment yet.

        Args:
            env_id: Env ID.

        Returns:
            The newly created EpisodeV2 instance.
        """
    assert env_id not in self._active_episodes
    new_episode = EpisodeV2(env_id, self._worker.policy_map, self._worker.policy_mapping_fn, worker=self._worker, callbacks=self._callbacks)
    self._callbacks.on_episode_created(worker=self._worker, base_env=self._base_env, policies=self._worker.policy_map, env_index=env_id, episode=new_episode)
    return new_episode