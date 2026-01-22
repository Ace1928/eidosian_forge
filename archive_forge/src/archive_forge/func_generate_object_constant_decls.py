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
def generate_object_constant_decls(self):
    consts = [(len(c.cname), c.cname, c) for c in self.py_constants]
    consts.sort()
    for _, cname, c in consts:
        self.parts['module_state'].putln('%s;' % c.type.declaration_code(cname))
        self.parts['module_state_defines'].putln('#define %s %s->%s' % (cname, Naming.modulestateglobal_cname, cname))
        if not c.type.needs_refcounting:
            continue
        self.parts['module_state_clear'].putln('Py_CLEAR(clear_module_state->%s);' % cname)
        self.parts['module_state_traverse'].putln('Py_VISIT(traverse_module_state->%s);' % cname)