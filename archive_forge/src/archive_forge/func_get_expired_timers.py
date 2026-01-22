import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
@abc.abstractmethod
def get_expired_timers(self, deadline: float) -> Dict[str, List[TimerRequest]]:
    """
        Returns all expired timers for each worker_id. An expired timer
        is a timer for which the expiration_time is less than or equal to
        the provided deadline.
        """
    pass