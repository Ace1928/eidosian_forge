import numpy as np
import logging
import gymnasium as gym
from typing import Dict, List, Optional, Type, Union
from ray.rllib.algorithms.impala import vtrace_tf as vtrace
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_bootstrap_value
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.policy.tf_mixins import GradStatsMixin, ValueNetworkMixin
from ray.rllib.utils.typing import (
@override(base)
def get_batch_divisibility_req(self) -> int:
    return self.config['rollout_fragment_length']