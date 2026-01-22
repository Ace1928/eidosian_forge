import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
@PublicAPI
def get_tf_eager_cls_if_necessary(orig_cls: Type['TFPolicy'], config: Union['AlgorithmConfig', PartialAlgorithmConfigDict]) -> Type[Union['TFPolicy', 'EagerTFPolicy', 'EagerTFPolicyV2']]:
    """Returns the corresponding tf-eager class for a given TFPolicy class.

    Args:
        orig_cls: The original TFPolicy class to get the corresponding tf-eager
            class for.
        config: The Algorithm config dict or AlgorithmConfig object.

    Returns:
        The tf eager policy class corresponding to the given TFPolicy class.
    """
    cls = orig_cls
    framework = config.get('framework', 'tf')
    if framework in ['tf2', 'tf'] and (not tf1):
        raise ImportError('Could not import tensorflow!')
    if framework == 'tf2':
        if not tf1.executing_eagerly():
            tf1.enable_eager_execution()
        assert tf1.executing_eagerly()
        from ray.rllib.policy.tf_policy import TFPolicy
        from ray.rllib.policy.eager_tf_policy import EagerTFPolicy
        from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
        if hasattr(orig_cls, 'as_eager') and (not issubclass(orig_cls, EagerTFPolicy)):
            cls = orig_cls.as_eager()
        elif not issubclass(orig_cls, TFPolicy):
            pass
        else:
            raise ValueError('This policy does not support eager execution: {}'.format(orig_cls))
        if config.get('eager_tracing') and issubclass(cls, (EagerTFPolicy, EagerTFPolicyV2)):
            cls = cls.with_tracing()
    return cls