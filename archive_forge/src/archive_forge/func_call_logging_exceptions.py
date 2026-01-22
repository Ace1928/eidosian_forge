from abc import ABC
import collections
import enum
import functools
import logging
def call_logging_exceptions(behavior, message, *args, **kwargs):
    """Calls a behavior in a try-except that logs any exceptions it raises.

    Args:
      behavior: Any callable.
      message: A string to log if the behavior raises an exception.
      *args: Positional arguments to pass to the given behavior.
      **kwargs: Keyword arguments to pass to the given behavior.

    Returns:
      An Outcome describing whether the given behavior returned a value or raised
        an exception.
    """
    return _call_logging_exceptions(behavior, message, *args, **kwargs)