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
def temps_holding_reference(self):
    """Return a list of (cname,type) tuples of temp names and their type
        that are currently in use. This includes only temps
        with a reference counted type which owns its reference.
        """
    return [(name, type) for name, type, manage_ref in self.temps_in_use() if manage_ref and type.needs_refcounting]