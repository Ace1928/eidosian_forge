import time
import logging
import typing as tp
from IPython.display import display
from copy import deepcopy
from typing import List, Optional, Any, Union
from .ipythonwidget import MetricWidget
def estimate_remaining_time(self, time_from_start: float) -> Optional[float]:
    if self.total_iterations is None:
        return None
    remaining_iterations: int = self.total_iterations - self.passed_iterations
    return time_from_start / self.passed_iterations * remaining_iterations