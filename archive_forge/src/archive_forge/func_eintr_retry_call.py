import errno
import os
import stat
import sys
from subprocess import check_output
def eintr_retry_call(func, *args, **kwargs):
    """
    Handle interruptions to an interruptible system call.

    Run an interruptible system call in a loop and retry if it raises EINTR.
    The signal calls that may raise EINTR prior to Python 3.5 are listed in
    PEP 0475.  Any calls to these functions must be wrapped in eintr_retry_call
    in order to handle EINTR returns in older versions of Python.

    This function is safe to use under Python 3.5 and newer since the wrapped
    function will simply return without raising EINTR.

    This function is based on _eintr_retry_call in python's subprocess.py.
    """
    import select
    while True:
        try:
            return func(*args, **kwargs)
        except (OSError, IOError, select.error) as err:
            if isinstance(err, (OSError, IOError)):
                error_code = err.errno
            else:
                error_code = err.args[0]
            if error_code == errno.EINTR:
                continue
            raise