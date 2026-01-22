import functools
import io
import logging
import os
import sys
import time
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from oslo_utils import timeutils
def forever_retry_uncaught_exceptions(*args, **kwargs):
    """Decorates provided function with infinite retry behavior.

    The function retry delay is **always** one second unless
    keyword argument ``retry_delay`` is passed that defines a value different
    than 1.0 (less than zero values are automatically changed to be 0.0).

    If repeated exceptions with the same message occur, logging will only
    output/get triggered for those equivalent messages every 60.0
    seconds, this can be altered by keyword argument ``same_log_delay`` to
    be a value different than 60.0 seconds (exceptions that change the
    message are always logged no matter what this delay is set to). As in
    the ``retry_delay`` case if this is less than zero, it will be
    automatically changed to be 0.0.
    """

    def decorator(infunc):
        retry_delay = max(0.0, float(kwargs.get('retry_delay', 1.0)))
        same_log_delay = max(0.0, float(kwargs.get('same_log_delay', 60.0)))

        @functools.wraps(infunc)
        def wrapper(*args, **kwargs):
            last_exc_message = None
            same_failure_count = 0
            watch = timeutils.StopWatch(duration=same_log_delay)
            while True:
                try:
                    return infunc(*args, **kwargs)
                except Exception as exc:
                    this_exc_message = encodeutils.exception_to_unicode(exc)
                    if this_exc_message == last_exc_message:
                        same_failure_count += 1
                    else:
                        same_failure_count = 1
                    if this_exc_message != last_exc_message or watch.expired():
                        logging.exception('Unexpected exception occurred %d time(s)... retrying.' % same_failure_count)
                        if not watch.has_started():
                            watch.start()
                        else:
                            watch.restart()
                        same_failure_count = 0
                        last_exc_message = this_exc_message
                    time.sleep(retry_delay)
        return wrapper
    if kwargs or not args:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    else:
        return decorator