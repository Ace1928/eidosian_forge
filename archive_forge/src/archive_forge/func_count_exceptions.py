import os
from threading import Lock
import time
import types
from typing import (
import warnings
from . import values  # retain this import style for testability
from .context_managers import ExceptionCounter, InprogressTracker, Timer
from .metrics_core import (
from .registry import Collector, CollectorRegistry, REGISTRY
from .samples import Exemplar, Sample
from .utils import floatToGoString, INF
def count_exceptions(self, exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]]=Exception) -> ExceptionCounter:
    """Count exceptions in a block of code or function.

        Can be used as a function decorator or context manager.
        Increments the counter when an exception of the given
        type is raised up out of the code.
        """
    self._raise_if_not_observable()
    return ExceptionCounter(self, exception)