import errno
import functools
import fcntl
import os
import struct
import threading
from . import exceptions
from . import _error_translation as errors
from .bindings import libzfs_core
from ._constants import MAXNAMELEN
from .ctypes import int32_t
from ._nvlist import nvlist_in, nvlist_out
def lzc_get_holds(snapname):
    """
    Retrieve list of *user holds* on the specified snapshot.

    :param bytes snapname: the name of the snapshot.
    :return: holds on the snapshot along with their creation times
             in seconds since the epoch
    :rtype: dict of bytes : int
    """
    holds = {}
    with nvlist_out(holds) as nvlist:
        ret = _lib.lzc_get_holds(snapname, nvlist)
    errors.lzc_get_holds_translate_error(ret, snapname)
    return holds