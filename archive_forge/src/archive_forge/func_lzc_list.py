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
@_uncommitted()
def lzc_list(name, options):
    """
    List subordinate elements of the given dataset.

    This function can be used to list child datasets and snapshots
    of the given dataset.  The listed elements can be filtered by
    their type and by their depth relative to the starting dataset.

    :param bytes name: the name of the dataset to be listed, could
                       be a snapshot or a dataset.
    :param options: a `dict` of the options that control the listing
                    behavior.
    :type options: dict of bytes:Any
    :return: a pair of file descriptors the first of which can be
             used to read the listing.
    :rtype: tuple of (int, int)
    :raises DatasetNotFound: if the dataset does not exist.

    Two options are currently available:

    recurse : integer or None
        specifies depth of the recursive listing. If ``None`` the
        depth is not limited.
        Absence of this option means that only the given dataset
        is listed.

    type : dict of bytes:None
        specifies dataset types to include into the listing.
        Currently allowed keys are "filesystem", "volume", "snapshot".
        Absence of this option implies all types.

    The first of the returned file descriptors can be used to
    read the listing in a binary encounded format.  The data is
    a series of variable sized records each starting with a fixed
    size header, the header is followed by a serialized ``nvlist``.
    Each record describes a single element and contains the element's
    name as well as its properties.
    The file descriptor must be closed after reading from it.

    The second file descriptor represents a pipe end to which the
    kernel driver is writing information.  It should not be closed
    until all interesting information has been read and it must
    be explicitly closed afterwards.
    """
    rfd, wfd = os.pipe()
    fcntl.fcntl(rfd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    fcntl.fcntl(wfd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    options = options.copy()
    options['fd'] = int32_t(wfd)
    opts_nv = nvlist_in(options)
    ret = _lib.lzc_list(name, opts_nv)
    if ret == errno.ESRCH:
        return (None, None)
    errors.lzc_list_translate_error(ret, name, options)
    return (rfd, wfd)