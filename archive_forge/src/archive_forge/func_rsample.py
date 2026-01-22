import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple
@override(Distribution)
def rsample(self, *, sample_shape: Tuple[int, ...]=None, **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
    rsamples = []
    for dist in self._flat_child_distributions:
        rsample = dist.rsample(sample_shape=sample_shape, **kwargs)
        rsamples.append(rsample)
    rsamples = tree.unflatten_as(self._original_struct, rsamples)
    return rsamples