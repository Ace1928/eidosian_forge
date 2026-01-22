from abc import ABCMeta, abstractmethod
from typing import Any, Union
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType

        Returns the tf-op that calculates the value based on a time step input.

        Args:
            t: The time step op (int tf.Tensor).

        Returns:
            The calculated value depending on the schedule and `t`.
        