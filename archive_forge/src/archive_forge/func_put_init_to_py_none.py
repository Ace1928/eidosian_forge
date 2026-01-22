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
def put_init_to_py_none(self, cname, type, nanny=True):
    from .PyrexTypes import py_object_type, typecast
    py_none = typecast(type, py_object_type, 'Py_None')
    if nanny:
        self.putln('%s = %s; __Pyx_INCREF(Py_None);' % (cname, py_none))
    else:
        self.putln('%s = %s; Py_INCREF(Py_None);' % (cname, py_none))