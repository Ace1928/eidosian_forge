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
def lzc_clone(name, origin, props=None):
    """
    Clone a ZFS filesystem or a ZFS volume ("zvol") from a given snapshot.

    :param bytes name: a name of the dataset to be created.
    :param bytes origin: a name of the origin snapshot.
    :param props: a `dict` of ZFS dataset property name-value pairs (empty by default).
    :type props: dict of bytes:Any

    :raises FilesystemExists: if a dataset with the given name already exists.
    :raises DatasetNotFound: if either a parent dataset of the requested dataset
                             or the origin snapshot does not exist.
    :raises PropertyInvalid: if one or more of the specified properties is invalid
                             or has an invalid type or value.
    :raises FilesystemNameInvalid: if the name is not a valid dataset name.
    :raises SnapshotNameInvalid: if the origin is not a valid snapshot name.
    :raises NameTooLong: if the name or the origin name is too long.
    :raises PoolsDiffer: if the clone and the origin have different pool names.

    .. note::
        Because of a deficiency of the underlying C interface
        :exc:`.DatasetNotFound` can mean that either a parent filesystem of the target
        or the origin snapshot does not exist.
        It is currently impossible to distinguish between the cases.
        :func:`lzc_hold` can be used to check that the snapshot exists and ensure that
        it is not destroyed before cloning.
    """
    if props is None:
        props = {}
    nvlist = nvlist_in(props)
    ret = _lib.lzc_clone(name, origin, nvlist)
    errors.lzc_clone_translate_error(ret, name, origin, props)