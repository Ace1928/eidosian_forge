from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
@property
def duration_s(self) -> str:
    """
        Returns the latest duration of the timer as a string
        """
    return self.pformat(self.get_duration(False))