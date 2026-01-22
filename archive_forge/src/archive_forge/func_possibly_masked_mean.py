import logging
from typing import Any, Dict, Mapping
from ray.rllib.algorithms.ppo.ppo_learner import (
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import TensorType
def possibly_masked_mean(t):
    return torch.sum(t[mask]) / num_valid