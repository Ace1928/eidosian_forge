import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_inherit_prop_translate_error(ret, name, prop):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_name(name)
        raise lzc_exc.PropertyInvalid(prop)
    if ret == errno.ENOENT:
        raise lzc_exc.DatasetNotFound(name)
    raise _generic_exception(ret, name, 'Failed to inherit a property')