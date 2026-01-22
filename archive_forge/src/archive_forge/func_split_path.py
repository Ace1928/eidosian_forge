import io
import os
import re
import tarfile
import tempfile
from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM
def split_path(p):
    return [pt for pt in re.split(_SEP, p) if pt and pt != '.']