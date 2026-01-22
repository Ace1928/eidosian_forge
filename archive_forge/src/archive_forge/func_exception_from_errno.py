import errno
import os
from ctypes import get_errno
def exception_from_errno(errnum):
    """Create an exception from ``errnum``.

    ``errnum`` is an integral error number.

    Return an exception object appropriate to ``errnum``.

    """
    exception = ERRNO_EXCEPTIONS.get(errnum)
    errorstr = os.strerror(errnum)
    if exception is None:
        return EnvironmentError(errnum, errorstr)
    return exception(errorstr)