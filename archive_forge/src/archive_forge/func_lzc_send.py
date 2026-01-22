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
def lzc_send(snapname, fromsnap, fd, flags=None):
    """
    Generate a zfs send stream for the specified snapshot and write it to
    the specified file descriptor.

    :param bytes snapname: the name of the snapshot to send.
    :param fromsnap: if not None the name of the starting snapshot
                     for the incremental stream.
    :type fromsnap: bytes or None
    :param int fd: the file descriptor to write the send stream to.
    :param flags: the flags that control what enhanced features can be used
                  in the stream.
    :type flags: list of bytes

    :raises SnapshotNotFound: if either the starting snapshot is not `None` and does not exist,
                              or if the ending snapshot does not exist.
    :raises NameInvalid: if the name of either snapshot is invalid.
    :raises NameTooLong: if the name of either snapshot is too long.
    :raises SnapshotMismatch: if ``fromsnap`` is not an ancestor snapshot of ``snapname``.
    :raises PoolsDiffer: if the snapshots belong to different pools.
    :raises IOError: if an input / output error occurs while writing to ``fd``.
    :raises UnknownStreamFeature: if the ``flags`` contain an unknown flag name.

    If ``fromsnap`` is None, a full (non-incremental) stream will be sent.
    If ``fromsnap`` is not None, it must be the full name of a snapshot or
    bookmark to send an incremental from, e.g. :file:`{pool}/{fs}@{earlier_snap}`
    or :file:`{pool}/{fs}#{earlier_bmark}`.

    The specified snapshot or bookmark must represent an earlier point in the history
    of ``snapname``.
    It can be an earlier snapshot in the same filesystem or zvol as ``snapname``,
    or it can be the origin of ``snapname``'s filesystem, or an earlier
    snapshot in the origin, etc.
    ``fromsnap`` must be strictly an earlier snapshot, specifying the same snapshot
    as both ``fromsnap`` and ``snapname`` is an error.

    If ``flags`` contains *"large_blocks"*, the stream is permitted
    to contain ``DRR_WRITE`` records with ``drr_length`` > 128K, and ``DRR_OBJECT``
    records with ``drr_blksz`` > 128K.

    If ``flags`` contains *"embedded_data"*, the stream is permitted
    to contain ``DRR_WRITE_EMBEDDED`` records with
    ``drr_etype`` == ``BP_EMBEDDED_TYPE_DATA``,
    which the receiving system must support (as indicated by support
    for the *embedded_data* feature).

    .. note::
        ``lzc_send`` can actually accept a filesystem name as the ``snapname``.
        In that case ``lzc_send`` acts as if a temporary snapshot was created
        after the start of the call and before the stream starts being produced.

    .. note::
        ``lzc_send`` does not return until all of the stream is written to ``fd``.

    .. note::
        ``lzc_send`` does *not* close ``fd`` upon returning.
    """
    if fromsnap is not None:
        c_fromsnap = fromsnap
    else:
        c_fromsnap = _ffi.NULL
    c_flags = 0
    if flags is None:
        flags = []
    for flag in flags:
        c_flag = {'embedded_data': _lib.LZC_SEND_FLAG_EMBED_DATA, 'large_blocks': _lib.LZC_SEND_FLAG_LARGE_BLOCK}.get(flag)
        if c_flag is None:
            raise exceptions.UnknownStreamFeature(flag)
        c_flags |= c_flag
    ret = _lib.lzc_send(snapname, c_fromsnap, fd, c_flags)
    errors.lzc_send_translate_error(ret, snapname, fromsnap, fd, flags)