import datetime
import errno
import os
import os.path
import time
def make_file_with_directories(path, private=False):
    """Creates a file and its containing directories, if they don't already
    exist.

    If `private` is True, the file will be made private (readable only by the
    current user) and so will the leaf directory. Pre-existing contents of the
    file are not modified.

    Passing `private=True` is not supported on Windows because it doesn't support
    the relevant parts of `os.chmod()`.

    Args:
      path: str, The path of the file to create.
      private: boolean, Whether to make the file and leaf directory readable only
        by the current user.

    Raises:
      RuntimeError: If called on Windows with `private` set to True.
    """
    if private and os.name == 'nt':
        raise RuntimeError('Creating private file not supported on Windows')
    try:
        path = os.path.realpath(path)
        leaf_dir = os.path.dirname(path)
        try:
            os.makedirs(leaf_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        if private:
            os.chmod(leaf_dir, 448)
        open(path, 'a').close()
        if private:
            os.chmod(path, 384)
    except EnvironmentError as e:
        raise RuntimeError('Failed to create file %s: %s' % (path, e))