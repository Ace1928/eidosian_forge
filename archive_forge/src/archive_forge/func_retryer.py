from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from apitools.base.py import http_wrapper as apitools_http_wrapper
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def retryer(target, should_retry_if, target_args=None, target_kwargs=None):
    """Retries the target with specific default value.

  This function is intended to be used for all gcloud storage's API requests
  that require custom retry handling (e.g downloads and uploads).

  Args:
    target (Callable): The function to call and retry.
    should_retry_if (Callable): func(exc_type, exc_value, exc_traceback, state)
        that returns True or False.
    target_args (Sequence|None): A sequence of positional arguments to be passed
        to the target.
    target_kwargs (Dict|None): A dict of keyword arguments to be passed
        to target.

  Returns:
    Whatever the target returns.
  """
    return retry.Retryer(max_retrials=properties.VALUES.storage.max_retries.GetInt(), wait_ceiling_ms=properties.VALUES.storage.max_retry_delay.GetInt() * 1000, exponential_sleep_multiplier=properties.VALUES.storage.exponential_sleep_multiplier.GetInt()).RetryOnException(target, args=target_args, kwargs=target_kwargs, sleep_ms=properties.VALUES.storage.base_retry_delay.GetInt() * 1000, should_retry_if=should_retry_if)