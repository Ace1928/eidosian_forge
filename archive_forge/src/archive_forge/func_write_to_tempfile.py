import contextlib
import errno
import hashlib
import json
import os
import stat
import tempfile
import time
import yaml
from oslo_utils import excutils
def write_to_tempfile(content, path=None, suffix='', prefix='tmp'):
    """Create a temporary file containing data.

    Create a temporary file containing specified content, with a specified
    filename suffix and prefix. The tempfile will be created in a default
    location, or in the directory `path`, if it is not None. `path` and its
    parent directories will be created if they don't exist.

    :param content: bytestring to write to the file
    :param path: same as parameter 'dir' for mkstemp
    :param suffix: same as parameter 'suffix' for mkstemp
    :param prefix: same as parameter 'prefix' for mkstemp

    For example: it can be used in database tests for creating
    configuration files.

    .. versionadded:: 1.9
    """
    if path:
        ensure_tree(path)
    fd, path = tempfile.mkstemp(suffix=suffix, dir=path, prefix=prefix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return path