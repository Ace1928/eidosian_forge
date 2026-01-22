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
def put_temp_declarations(self, func_context):
    for name, type, manage_ref, static in func_context.temps_allocated:
        if type.is_cpp_class and (not type.is_fake_reference) and func_context.scope.directives['cpp_locals']:
            decl = type.cpp_optional_declaration_code(name)
        else:
            decl = type.declaration_code(name)
        if type.is_pyobject:
            self.putln('%s = NULL;' % decl)
        elif type.is_memoryviewslice:
            self.putln('%s = %s;' % (decl, type.literal_code(type.default_value)))
        else:
            self.putln('%s%s;' % (static and 'static ' or '', decl))
    if func_context.should_declare_error_indicator:
        if self.funcstate.uses_error_indicator:
            unused = ''
        else:
            unused = 'CYTHON_UNUSED '
        self.putln('%sint %s = 0;' % (unused, Naming.lineno_cname))
        self.putln('%sconst char *%s = NULL;' % (unused, Naming.filename_cname))
        self.putln('%sint %s = 0;' % (unused, Naming.clineno_cname))