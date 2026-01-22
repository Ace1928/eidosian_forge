from __future__ import (absolute_import, division, print_function)
import copy
import functools
import itertools
import random
import sys
import time
import ansible.module_utils.compat.typing as t
def retry_with_delays_and_condition(backoff_iterator, should_retry_error=None):
    """Generic retry decorator.

    :param backoff_iterator: An iterable of delays in seconds.
    :param should_retry_error: A callable that takes an exception of the decorated function and decides whether to retry or not (returns a bool).
    """

    def _emit_isolated_iterator_copies(original_iterator):
        _copiable_iterator, _first_iterator_copy = itertools.tee(original_iterator)
        yield _first_iterator_copy
        while True:
            yield copy.copy(_copiable_iterator)
    backoff_iterator_generator = _emit_isolated_iterator_copies(backoff_iterator)
    del backoff_iterator
    if should_retry_error is None:
        should_retry_error = retry_never

    def function_wrapper(function):

        @functools.wraps(function)
        def run_function(*args, **kwargs):
            """This assumes the function has not already been called.
            If backoff_iterator is empty, we should still run the function a single time with no delay.
            """
            call_retryable_function = functools.partial(function, *args, **kwargs)
            for delay in next(backoff_iterator_generator):
                try:
                    return call_retryable_function()
                except Exception as e:
                    if not should_retry_error(e):
                        raise
                time.sleep(delay)
            return call_retryable_function()
        return run_function
    return function_wrapper