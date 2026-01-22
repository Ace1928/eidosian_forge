from __future__ import annotations
from typing import (
import sys
import time
import functools
from google.api_core.retry.retry_base import _BaseRetry
from google.api_core.retry.retry_base import _retry_error_helper
from google.api_core.retry import exponential_sleep_generator
from google.api_core.retry import build_retry_error
from google.api_core.retry import RetryFailureReason
@functools.wraps(func)
def retry_wrapped_func(*args: _P.args, **kwargs: _P.kwargs) -> Generator[_Y, Any, None]:
    """A wrapper that calls target function with retry."""
    sleep_generator = exponential_sleep_generator(self._initial, self._maximum, multiplier=self._multiplier)
    return retry_target_stream(func, predicate=self._predicate, sleep_generator=sleep_generator, timeout=self._timeout, on_error=on_error, init_args=args, init_kwargs=kwargs)