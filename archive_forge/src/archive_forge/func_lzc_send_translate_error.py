import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_send_translate_error(ret, snapname, fromsnap, fd, flags):
    if ret == 0:
        return
    if ret == errno.EXDEV and fromsnap is not None:
        if _pool_name(fromsnap) != _pool_name(snapname):
            raise lzc_exc.PoolsDiffer(snapname)
        else:
            raise lzc_exc.SnapshotMismatch(snapname)
    elif ret == errno.EINVAL:
        if fromsnap is not None and (not _is_valid_snap_name(fromsnap)) and (not _is_valid_bmark_name(fromsnap)):
            raise lzc_exc.NameInvalid(fromsnap)
        elif not _is_valid_snap_name(snapname) and (not _is_valid_fs_name(snapname)):
            raise lzc_exc.NameInvalid(snapname)
        elif fromsnap is not None and len(fromsnap) > MAXNAMELEN:
            raise lzc_exc.NameTooLong(fromsnap)
        elif len(snapname) > MAXNAMELEN:
            raise lzc_exc.NameTooLong(snapname)
        elif fromsnap is not None and _pool_name(fromsnap) != _pool_name(snapname):
            raise lzc_exc.PoolsDiffer(snapname)
    elif ret == errno.ENOENT:
        if fromsnap is not None and (not _is_valid_snap_name(fromsnap)) and (not _is_valid_bmark_name(fromsnap)):
            raise lzc_exc.NameInvalid(fromsnap)
        raise lzc_exc.SnapshotNotFound(snapname)
    elif ret == errno.ENAMETOOLONG:
        if fromsnap is not None and len(fromsnap) > MAXNAMELEN:
            raise lzc_exc.NameTooLong(fromsnap)
        else:
            raise lzc_exc.NameTooLong(snapname)
    raise lzc_exc.StreamIOError(ret)