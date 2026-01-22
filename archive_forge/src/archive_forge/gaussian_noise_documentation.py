from gymnasium.spaces import Space
import numpy as np
from typing import Union, Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import (
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import Schedule
from ray.rllib.utils.schedules.piecewise_schedule import PiecewiseSchedule
from ray.rllib.utils.tf_utils import zero_logps_from_actions
Returns the current scale value.

        Returns:
            Union[float,tf.Tensor[float]]: The current scale value.
        