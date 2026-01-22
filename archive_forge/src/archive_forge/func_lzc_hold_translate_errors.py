import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_hold_translate_errors(ret, errlist, holds, fd):
    if ret == 0:
        return

    def _map(ret, name):
        if ret == errno.EXDEV:
            return lzc_exc.PoolsDiffer(name)
        elif ret == errno.EINVAL:
            if name:
                pool_names = map(_pool_name, holds.keys())
                if not _is_valid_snap_name(name):
                    return lzc_exc.NameInvalid(name)
                elif len(name) > MAXNAMELEN:
                    return lzc_exc.NameTooLong(name)
                elif any((x != _pool_name(name) for x in pool_names)):
                    return lzc_exc.PoolsDiffer(name)
            else:
                invalid_names = [b for b in holds.keys() if not _is_valid_snap_name(b)]
                if invalid_names:
                    return lzc_exc.NameInvalid(invalid_names[0])
        fs_name = None
        hold_name = None
        pool_name = None
        if name is not None:
            fs_name = _fs_name(name)
            pool_name = _pool_name(name)
            hold_name = holds[name]
        if ret == errno.ENOENT:
            return lzc_exc.FilesystemNotFound(fs_name)
        if ret == errno.EEXIST:
            return lzc_exc.HoldExists(name)
        if ret == errno.E2BIG:
            return lzc_exc.NameTooLong(hold_name)
        if ret == errno.ENOTSUP:
            return lzc_exc.FeatureNotSupported(pool_name)
        return _generic_exception(ret, name, 'Failed to hold snapshot')
    if ret == errno.EBADF:
        raise lzc_exc.BadHoldCleanupFD()
    _handle_err_list(ret, errlist, holds.keys(), lzc_exc.HoldFailure, _map)