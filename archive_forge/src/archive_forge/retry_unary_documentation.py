from __future__ import annotations
import functools
import sys
import time
import inspect
import warnings
from typing import Any, Callable, Iterable, TypeVar, TYPE_CHECKING
from google.api_core.retry.retry_base import _BaseRetry
from google.api_core.retry.retry_base import _retry_error_helper
from google.api_core.retry.retry_base import exponential_sleep_generator
from google.api_core.retry.retry_base import build_retry_error
from google.api_core.retry.retry_base import RetryFailureReason
A wrapper that calls target function with retry.