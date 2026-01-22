import gymnasium as gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.policy import eager_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import (
Allows creating a TFPolicy cls based on settings of another one.

        Keyword Args:
            **overrides: The settings (passed into `build_tf_policy`) that
                should be different from the class that this method is called
                on.

        Returns:
            type: A new TFPolicy sub-class.

        Examples:
        >> MySpecialDQNPolicyClass = DQNTFPolicy.with_updates(
        ..    name="MySpecialDQNPolicyClass",
        ..    loss_function=[some_new_loss_function],
        .. )
        