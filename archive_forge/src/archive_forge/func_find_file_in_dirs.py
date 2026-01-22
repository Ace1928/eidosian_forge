import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def find_file_in_dirs(path, dirs):
    """
    Search for `path` in the list of directories `dirs`.

    Return the first expansion that matches an existing file.
    """
    if os.path.isabs(path):
        return path
    for d in dirs:
        if d == '.':
            f = path
        else:
            d = os.path.expanduser(d)
            f = os.path.join(d, path)
        if os.path.exists(f):
            return f
    return path