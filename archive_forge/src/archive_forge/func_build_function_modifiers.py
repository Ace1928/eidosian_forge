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
def build_function_modifiers(self, modifiers, mapper=modifier_output_mapper):
    if not modifiers:
        return ''
    return '%s ' % ' '.join([mapper(m, m) for m in modifiers])