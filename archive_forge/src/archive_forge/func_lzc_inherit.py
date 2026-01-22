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
def lzc_inherit(name, prop):
    """
    Inherit properties from a parent dataset of the given ZFS dataset.

    :param bytes name: the name of the dataset.
    :param bytes prop: the name of the property to inherit.
    :raises NameInvalid: if the dataset name is invalid.
    :raises NameTooLong: if the dataset name is too long.
    :raises DatasetNotFound: if the dataset does not exist.
    :raises PropertyInvalid: if one or more of the specified properties is invalid
                             or has an invalid type or value.

    Inheriting a property actually resets it to its default value
    or removes it if it's a user property, so that the property could be
    inherited if it's inheritable.  If the property is not inheritable
    then it would just have its default value.

    This function can be used on snapshots to inherit user defined properties.
    """
    ret = _lib.lzc_inherit(name, prop, _ffi.NULL)
    errors.lzc_inherit_prop_translate_error(ret, name, prop)