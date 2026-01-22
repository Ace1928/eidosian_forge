import logging
from typing import Type, Dict, Any, Optional, Union
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
def get_rollout_fragment_length(self, worker_index: int=0) -> int:
    if self.rollout_fragment_length == 'auto':
        return self.n_step
    else:
        return self.rollout_fragment_length