from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
def read_utilities_from_utility_dir(path):
    """
    Read all lines of the file at the provided path from a path relative
    to get_utility_dir().
    """
    filename = os.path.join(get_utility_dir(), path)
    with closing(Utils.open_source_file(filename, encoding='UTF-8')) as f:
        return f.readlines()