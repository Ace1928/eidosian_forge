import logging
import os
import sys
def get_path_uid(path: str) -> int:
    """
    Return path's uid.

    Does not follow symlinks:
        https://github.com/pypa/pip/pull/935#discussion_r5307003

    Placed this function in compat due to differences on AIX and
    Jython, that should eventually go away.

    :raises OSError: When path is a symlink or can't be read.
    """
    if hasattr(os, 'O_NOFOLLOW'):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        file_uid = os.fstat(fd).st_uid
        os.close(fd)
    elif not os.path.islink(path):
        file_uid = os.stat(path).st_uid
    else:
        raise OSError(f'{path} is a symlink; Will not return uid for symlinks')
    return file_uid