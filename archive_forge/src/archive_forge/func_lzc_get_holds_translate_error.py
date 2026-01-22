import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_get_holds_translate_error(ret, snapname):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_snap_name(snapname)
    if ret == errno.ENOENT:
        raise lzc_exc.SnapshotNotFound(snapname)
    if ret == errno.ENOTSUP:
        raise lzc_exc.FeatureNotSupported(_pool_name(snapname))
    raise _generic_exception(ret, snapname, 'Failed to get holds on snapshot')