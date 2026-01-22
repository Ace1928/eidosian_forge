from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def with_delay(self, initial: Optional[float]=None, maximum: Optional[float]=None, multiplier: Optional[float]=None) -> Self:
    """Return a copy of this retry with the given delay options.

        Args:
            initial (float): The minimum amount of time to delay (in seconds). This must
                be greater than 0. If None, the current value is used.
            maximum (float): The maximum amount of time to delay (in seconds). If None, the
                current value is used.
            multiplier (float): The multiplier applied to the delay. If None, the current
                value is used.

        Returns:
            Retry: A new retry instance with the given delay options.
        """
    return type(self)(predicate=self._predicate, initial=initial if initial is not None else self._initial, maximum=maximum if maximum is not None else self._maximum, multiplier=multiplier if multiplier is not None else self._multiplier, timeout=self._timeout, on_error=self._on_error)