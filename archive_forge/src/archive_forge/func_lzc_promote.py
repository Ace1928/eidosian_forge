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
def lzc_promote(name):
    """
    Promotes the ZFS dataset.

    :param bytes name: the name of the dataset to promote.
    :raises NameInvalid: if the dataset name is invalid.
    :raises NameTooLong: if the dataset name is too long.
    :raises NameTooLong: if the dataset's origin has a snapshot that,
                         if transferred to the dataset, would get
                         a too long name.
    :raises NotClone: if the dataset is not a clone.
    :raises FilesystemNotFound: if the dataset does not exist.
    :raises SnapshotExists: if the dataset already has a snapshot with
                            the same name as one of the origin's snapshots.
    """
    ret = _lib.lzc_promote(name, _ffi.NULL, _ffi.NULL)
    errors.lzc_promote_translate_error(ret, name)