from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def with_timeout(self, timeout: float | None) -> Self:
    """Return a copy of this retry with the given timeout.

        Args:
            timeout (float): How long to keep retrying, in seconds. If None,
                no timeout will be enforced.

        Returns:
            Retry: A new retry instance with the given timeout.
        """
    return type(self)(predicate=self._predicate, initial=self._initial, maximum=self._maximum, multiplier=self._multiplier, timeout=timeout, on_error=self._on_error)