from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def elapsed_average(self, count: int) -> float:
    """
        Returns the average duration of the timer
        """
    return self.elapsed / count