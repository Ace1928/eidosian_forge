import abc
from copy import deepcopy
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Type
from ray.rllib.utils import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import TensorType
def rdrop(self, n: int) -> 'TensorSpec':
    """Drops the last n dimensions.

        Returns a copy of this TensorSpec with the rightmost n dimensions removed.

        Args:
            n: A positive number of dimensions to remove from the right

        Returns:
            A copy of the tensor spec with the last n dims removed

        Raises:
            IndexError: If n is greater than the number of indices in self
            AssertionError: If n is negative or not an int
        """
    assert isinstance(n, int) and n >= 0, 'n must be a positive integer or zero'
    copy_ = deepcopy(self)
    copy_._expected_shape = copy_.shape[:-n]
    copy_._full_shape = self._get_full_shape()
    return copy_