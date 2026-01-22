from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def total_average_s(self, count: int) -> str:
    """
        Returns the average duration of the timer as a string
        """
    return self.pformat(self.total_average(count))