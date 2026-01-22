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
def put_pymethoddef_wrapper(self, entry):
    func_cname = entry.func_cname
    if entry.is_special:
        method_flags = entry.signature.method_flags() or []
        from .TypeSlots import method_noargs
        if method_noargs in method_flags:
            func_cname = Naming.method_wrapper_prefix + func_cname
            self.putln('static PyObject *%s(PyObject *self, CYTHON_UNUSED PyObject *arg) {' % func_cname)
            func_call = '%s(self)' % entry.func_cname
            if entry.name == '__next__':
                self.putln('PyObject *res = %s;' % func_call)
                self.putln('if (!res && !PyErr_Occurred()) { PyErr_SetNone(PyExc_StopIteration); }')
                self.putln('return res;')
            else:
                self.putln('return %s;' % func_call)
            self.putln('}')
    return func_cname