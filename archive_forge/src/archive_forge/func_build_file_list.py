import io
import os
import re
import tarfile
import tempfile
from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM
def build_file_list(root):
    files = []
    for dirname, dirnames, fnames in os.walk(root):
        for filename in fnames + dirnames:
            longpath = os.path.join(dirname, filename)
            files.append(longpath.replace(root, '', 1).lstrip('/'))
    return files