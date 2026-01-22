from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def frombase(path1, path2):
    """Get the final path of ``path2`` that isn't in ``path1``.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        str: the final part of ``path2``.

    Example:
        >>> frombase('foo/bar/', 'foo/bar/baz/egg')
        'baz/egg'

    """
    if not isparent(path1, path2):
        raise ValueError('path1 must be a prefix of path2')
    return path2[len(path1):]