import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_snaprange_space_translate_error(ret, firstsnap, lastsnap):
    if ret == 0:
        return
    if ret == errno.EXDEV and firstsnap is not None:
        if _pool_name(firstsnap) != _pool_name(lastsnap):
            raise lzc_exc.PoolsDiffer(lastsnap)
        else:
            raise lzc_exc.SnapshotMismatch(lastsnap)
    if ret == errno.EINVAL:
        if not _is_valid_snap_name(firstsnap):
            raise lzc_exc.NameInvalid(firstsnap)
        elif not _is_valid_snap_name(lastsnap):
            raise lzc_exc.NameInvalid(lastsnap)
        elif len(firstsnap) > MAXNAMELEN:
            raise lzc_exc.NameTooLong(firstsnap)
        elif len(lastsnap) > MAXNAMELEN:
            raise lzc_exc.NameTooLong(lastsnap)
        elif _pool_name(firstsnap) != _pool_name(lastsnap):
            raise lzc_exc.PoolsDiffer(lastsnap)
        else:
            raise lzc_exc.SnapshotMismatch(lastsnap)
    if ret == errno.ENOENT:
        raise lzc_exc.SnapshotNotFound(lastsnap)
    raise _generic_exception(ret, lastsnap, 'Failed to calculate space used by range of snapshots')