import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_list_translate_error(ret, name, opts):
    if ret == 0:
        return
    if ret == errno.ENOENT:
        raise lzc_exc.DatasetNotFound(name)
    if ret == errno.EINVAL:
        _validate_fs_or_snap_name(name)
    raise _generic_exception(ret, name, 'Error obtaining a list')