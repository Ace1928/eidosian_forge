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
def inject_string_constants(self, impl, output):
    """Replace 'PYIDENT("xyz")' by a constant Python identifier cname.
        """
    if 'PYIDENT(' not in impl and 'PYUNICODE(' not in impl:
        return (False, impl)
    replacements = {}

    def externalise(matchobj):
        key = matchobj.groups()
        try:
            cname = replacements[key]
        except KeyError:
            str_type, name = key
            cname = replacements[key] = output.get_py_string_const(StringEncoding.EncodedString(name), identifier=str_type == 'IDENT').cname
        return cname
    impl = re.sub('PY(IDENT|UNICODE)\\("([^"]+)"\\)', externalise, impl)
    assert 'PYIDENT(' not in impl and 'PYUNICODE(' not in impl
    return (True, impl)