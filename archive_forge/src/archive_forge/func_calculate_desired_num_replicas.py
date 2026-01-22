import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional
from ray.serve._private.common import TargetCapacityDirection
from ray.serve._private.constants import CONTROL_LOOP_PERIOD_S, SERVE_LOGGER_NAME
from ray.serve._private.utils import get_capacity_adjusted_num_replicas
from ray.serve.config import AutoscalingConfig
def calculate_desired_num_replicas(autoscaling_config: AutoscalingConfig, current_num_ongoing_requests: List[float], override_min_replicas: Optional[float]=None, override_max_replicas: Optional[float]=None) -> int:
    """Returns the number of replicas to scale to based on the given metrics.

    Args:
        autoscaling_config: The autoscaling parameters to use for this
            calculation.
        current_num_ongoing_requests (List[float]): A list of the number of
            ongoing requests for each replica.  Assumes each entry has already
            been time-averaged over the desired lookback window.
        override_min_replicas: Overrides min_replicas from the config
            when calculating the final number of replicas.
        override_max_replicas: Overrides max_replicas from the config
            when calculating the final number of replicas.

    Returns:
        desired_num_replicas: The desired number of replicas to scale to, based
            on the input metrics and the current number of replicas.

    """
    current_num_replicas = len(current_num_ongoing_requests)
    if current_num_replicas == 0:
        raise ValueError('Number of replicas cannot be zero')
    num_ongoing_requests_per_replica: float = sum(current_num_ongoing_requests) / len(current_num_ongoing_requests)
    error_ratio: float = num_ongoing_requests_per_replica / autoscaling_config.target_num_ongoing_requests_per_replica
    if error_ratio >= 1:
        smoothing_factor = autoscaling_config.get_upscale_smoothing_factor()
    else:
        smoothing_factor = autoscaling_config.get_downscale_smoothing_factor()
    smoothed_error_ratio = 1 + (error_ratio - 1) * smoothing_factor
    desired_num_replicas = math.ceil(current_num_replicas * smoothed_error_ratio)
    if error_ratio == 0 and desired_num_replicas == current_num_replicas and (desired_num_replicas >= 1):
        desired_num_replicas -= 1
    min_replicas = autoscaling_config.min_replicas
    max_replicas = autoscaling_config.max_replicas
    if override_min_replicas is not None:
        min_replicas = override_min_replicas
    if override_max_replicas is not None:
        max_replicas = override_max_replicas
    desired_num_replicas = max(min_replicas, min(max_replicas, desired_num_replicas))
    return desired_num_replicas