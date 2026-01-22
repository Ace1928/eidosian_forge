import os
from shutil import copymode, copystat
from django.core.files import locks
def file_move_safe(old_file_name, new_file_name, chunk_size=1024 * 64, allow_overwrite=False):
    """
    Move a file from one location to another in the safest way possible.

    First, try ``os.rename``, which is simple but will break across filesystems.
    If that fails, stream manually from one file to another in pure Python.

    If the destination file exists and ``allow_overwrite`` is ``False``, raise
    ``FileExistsError``.
    """
    if _samefile(old_file_name, new_file_name):
        return
    try:
        if not allow_overwrite and os.access(new_file_name, os.F_OK):
            raise FileExistsError('Destination file %s exists and allow_overwrite is False.' % new_file_name)
        os.rename(old_file_name, new_file_name)
        return
    except OSError:
        pass
    with open(old_file_name, 'rb') as old_file:
        fd = os.open(new_file_name, os.O_WRONLY | os.O_CREAT | getattr(os, 'O_BINARY', 0) | (os.O_EXCL if not allow_overwrite else 0))
        try:
            locks.lock(fd, locks.LOCK_EX)
            current_chunk = None
            while current_chunk != b'':
                current_chunk = old_file.read(chunk_size)
                os.write(fd, current_chunk)
        finally:
            locks.unlock(fd)
            os.close(fd)
    try:
        copystat(old_file_name, new_file_name)
    except PermissionError:
        try:
            copymode(old_file_name, new_file_name)
        except PermissionError:
            pass
    try:
        os.remove(old_file_name)
    except PermissionError as e:
        if getattr(e, 'winerror', 0) != 32:
            raise