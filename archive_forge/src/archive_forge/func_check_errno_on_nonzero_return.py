import errno
import os
from ctypes import get_errno
def check_errno_on_nonzero_return(result, _func, *_args):
    """Error checker to check the system ``errno`` as returned by
    :func:`ctypes.get_errno()`.

    If ``result`` is not ``0``, an exception according to this errno is raised.
    Otherwise nothing happens.

    """
    if result != 0:
        errnum = get_errno()
        if errnum != 0:
            raise exception_from_errno(errnum)
    return result