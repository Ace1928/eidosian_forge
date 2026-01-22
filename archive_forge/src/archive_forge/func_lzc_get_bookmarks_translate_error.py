import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_get_bookmarks_translate_error(ret, fsname, props):
    if ret == 0:
        return
    if ret == errno.ENOENT:
        raise lzc_exc.FilesystemNotFound(fsname)
    raise _generic_exception(ret, fsname, 'Failed to list bookmarks')