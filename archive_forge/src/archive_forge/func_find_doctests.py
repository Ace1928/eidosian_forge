from __future__ import print_function
import atexit
import doctest
import os
from pkg_resources import (
def find_doctests(suffix):
    """Find doctests matching a certain suffix."""
    doctest_files = []
    if resource_exists('lazr.uri', 'docs'):
        for name in resource_listdir('lazr.uri', 'docs'):
            if name.endswith(suffix):
                doctest_files.append(os.path.abspath(resource_filename('lazr.uri', 'docs/%s' % name)))
    return doctest_files