import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional
from ray.serve._private.common import TargetCapacityDirection
from ray.serve._private.constants import CONTROL_LOOP_PERIOD_S, SERVE_LOGGER_NAME
from ray.serve._private.utils import get_capacity_adjusted_num_replicas
from ray.serve.config import AutoscalingConfig
def get_decision_num_replicas(self, curr_target_num_replicas: int, current_num_ongoing_requests: List[float], current_handle_queued_queries: float, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> int:
    if len(current_num_ongoing_requests) == 0:
        if current_handle_queued_queries > 0:
            return max(math.ceil(1 * self.config.get_upscale_smoothing_factor()), curr_target_num_replicas)
        return curr_target_num_replicas
    decision_num_replicas = curr_target_num_replicas
    desired_num_replicas = calculate_desired_num_replicas(self.config, current_num_ongoing_requests, override_min_replicas=self.get_current_lower_bound(target_capacity, target_capacity_direction), override_max_replicas=get_capacity_adjusted_num_replicas(self.config.max_replicas, target_capacity))
    if desired_num_replicas > curr_target_num_replicas:
        if self.decision_counter < 0:
            self.decision_counter = 0
        self.decision_counter += 1
        if self.decision_counter > self.scale_up_consecutive_periods:
            self.decision_counter = 0
            decision_num_replicas = desired_num_replicas
    elif desired_num_replicas < curr_target_num_replicas:
        if self.decision_counter > 0:
            self.decision_counter = 0
        self.decision_counter -= 1
        if self.decision_counter < -self.scale_down_consecutive_periods:
            self.decision_counter = 0
            decision_num_replicas = desired_num_replicas
    else:
        self.decision_counter = 0
    return decision_num_replicas