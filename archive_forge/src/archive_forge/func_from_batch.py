from collections import OrderedDict
import contextlib
import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.preprocessors import get_preprocessor, RepeatedValuesPreprocessor
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf, try_import_torch, TensorType
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict, TensorStructType
@Deprecated(new='ModelV2.__call__()', error=True)
def from_batch(self, train_batch: SampleBatch, is_training: bool=True) -> (TensorType, List[TensorType]):
    """Convenience function that calls this model with a tensor batch.

        All this does is unpack the tensor batch to call this model with the
        right input dict, state, and seq len arguments.
        """
    input_dict = train_batch.copy()
    input_dict.set_training(is_training)
    states = []
    i = 0
    while 'state_in_{}'.format(i) in input_dict:
        states.append(input_dict['state_in_{}'.format(i)])
        i += 1
    ret = self.__call__(input_dict, states, input_dict.get(SampleBatch.SEQ_LENS))
    return ret