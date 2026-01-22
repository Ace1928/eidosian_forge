from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def declare_arg(self, name, type, pos):
    name = self.mangle_class_private_name(name)
    cname = self.mangle(Naming.var_prefix, name)
    entry = self.declare(name, cname, type, pos, 'private')
    entry.is_variable = 1
    if type.is_pyobject:
        entry.init = '0'
    entry.is_arg = 1
    self.arg_entries.append(entry)
    return entry