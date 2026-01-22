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
@_uncommitted(lzc_list)
def lzc_get_props(name):
    """
    Get properties of the ZFS dataset.

    :param bytes name: the name of the dataset.
    :raises DatasetNotFound: if the dataset does not exist.
    :raises NameInvalid: if the dataset name is invalid.
    :raises NameTooLong: if the dataset name is too long.
    :return: a dictionary mapping the property names to their values.
    :rtype: dict of bytes:Any

    .. note::
        The value of ``clones`` property is a `list` of clone names
        as byte strings.

    .. warning::
        The returned dictionary does not contain entries for properties
        with default values.  One exception is the ``mountpoint`` property
        for which the default value is derived from the dataset name.
    """
    result = next(_list(name, recurse=0))
    is_snapshot = result['dmu_objset_stats']['dds_is_snapshot']
    result = result['properties']
    mountpoint = result.get('mountpoint')
    if mountpoint is not None:
        mountpoint_src = mountpoint['source']
        mountpoint_val = mountpoint['value']
        if mountpoint_val.startswith('/') and (not mountpoint_src.startswith('$')):
            mountpoint_val = mountpoint_val + name[len(mountpoint_src):]
    elif not is_snapshot:
        mountpoint_val = '/' + name
    else:
        mountpoint_val = None
    result = {k: v['value'] for k, v in result.iteritems()}
    if 'clones' in result:
        result['clones'] = result['clones'].keys()
    if mountpoint_val is not None:
        result['mountpoint'] = mountpoint_val
    return result