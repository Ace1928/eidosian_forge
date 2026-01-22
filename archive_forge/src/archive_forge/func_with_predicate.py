from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def with_predicate(self, predicate: Callable[[Exception], bool]) -> Self:
    """Return a copy of this retry with the given predicate.

        Args:
            predicate (Callable[Exception]): A callable that should return
                ``True`` if the given exception is retryable.

        Returns:
            Retry: A new retry instance with the given predicate.
        """
    return type(self)(predicate=predicate, initial=self._initial, maximum=self._maximum, multiplier=self._multiplier, timeout=self._timeout, on_error=self._on_error)