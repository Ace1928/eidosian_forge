from __future__ import print_function, unicode_literals
import sys
import typing
from fs.path import abspath, join, normpath
def sort_key_dirs_first(info):
    """Get the info sort function with directories first."""
    return (not info.is_dir, info.name.lower())