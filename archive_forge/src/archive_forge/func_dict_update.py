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
def dict_update(self, d, key):
    """
        Insert `self` in dict `d` with key `key`. If that key already
        exists, update the attributes of the existing value with `self`.
        """
    if key in d:
        other = d[key]
        other.location = min(self.location, other.location)
        other.pieces.update(self.pieces)
    else:
        d[key] = self