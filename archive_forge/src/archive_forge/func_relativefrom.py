from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def relativefrom(base, path):
    """Return a path relative from a given base path.

    Insert backrefs as appropriate to reach the path from the base.

    Arguments:
        base (str): Path to a directory.
        path (str): Path to make relative.

    Returns:
        str: the path to ``base`` from ``path``.

    >>> relativefrom("foo/bar", "baz/index.html")
    '../../baz/index.html'

    """
    base_parts = list(iteratepath(base))
    path_parts = list(iteratepath(path))
    common = 0
    for component_a, component_b in zip(base_parts, path_parts):
        if component_a != component_b:
            break
        common += 1
    return '/'.join(['..'] * (len(base_parts) - common) + path_parts[common:])