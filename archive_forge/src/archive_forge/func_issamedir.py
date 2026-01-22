from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def issamedir(path1, path2):
    """Check if two paths reference a resource in the same directory.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        bool: `True` if the two resources are in the same directory.

    Example:
        >>> issamedir("foo/bar/baz.txt", "foo/bar/spam.txt")
        True
        >>> issamedir("foo/bar/baz/txt", "spam/eggs/spam.txt")
        False

    """
    return dirname(normpath(path1)) == dirname(normpath(path2))