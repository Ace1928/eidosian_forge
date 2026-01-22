import logging
import random
import sys
import time
import traceback
from google.cloud.ml.util import _exceptions
from six import reraise
def with_exponential_backoff(num_retries=10, initial_delay_secs=1, logger=logging.warning, retry_filter=retry_on_server_errors_filter, clock=Clock(), fuzz=True):
    """Decorator with arguments that control the retry logic.

  Args:
    num_retries: The total number of times to retry.
    initial_delay_secs: The delay before the first retry, in seconds.
    logger: A callable used to report en exception. Must have the same signature
      as functions in the standard logging module. The default is
      logging.warning.
    retry_filter: A callable getting the exception raised and returning True
      if the retry should happen. For instance we do not want to retry on
      404 Http errors most of the time. The default value will return true
      for server errors (HTTP status code >= 500) and non Http errors.
    clock: A clock object implementing a sleep method. The default clock will
      use time.sleep().
    fuzz: True if the delay should be fuzzed (default). During testing False
      can be used so that the delays are not randomized.

  Returns:
    As per Python decorators with arguments pattern returns a decorator
    for the function which in turn will return the wrapped (decorated) function.

  The decorator is intended to be used on callables that make HTTP or RPC
  requests that can temporarily timeout or have transient errors. For instance
  the make_http_request() call below will be retried 16 times with exponential
  backoff and fuzzing of the delay interval (default settings).

  from cloudml.util import retry
  # ...
  @retry.with_exponential_backoff()
  make_http_request(args)
  """

    def real_decorator(fun):
        """The real decorator whose purpose is to return the wrapped function."""
        retry_intervals = iter(FuzzedExponentialIntervals(initial_delay_secs, num_retries, fuzz=0.5 if fuzz else 0))

        def wrapper(*args, **kwargs):
            while True:
                try:
                    return fun(*args, **kwargs)
                except Exception as exn:
                    if not retry_filter(exn):
                        raise
                    exn_traceback = sys.exc_info()[2]
                    try:
                        try:
                            sleep_interval = next(retry_intervals)
                        except StopIteration:
                            reraise(type(exn), exn, exn_traceback)
                        logger('Retry with exponential backoff: waiting for %s seconds before retrying %s because we caught exception: %s Traceback for above exception (most recent call last):\n%s', sleep_interval, getattr(fun, '__name__', str(fun)), ''.join(traceback.format_exception_only(exn.__class__, exn)), ''.join(traceback.format_tb(exn_traceback)))
                        clock.sleep(sleep_interval)
                    finally:
                        if sys.version_info < (3, 0):
                            sys.exc_clear()
                        exn_traceback = None
        return wrapper
    return real_decorator