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
def declare_module(self, name, scope, pos):
    entry = self.lookup_here(name)
    if entry:
        if entry.is_pyglobal and entry.as_module is scope:
            return entry
        if not (entry.is_pyglobal and (not entry.as_module)):
            return entry
    else:
        entry = self.declare_var(name, py_object_type, pos)
        entry.is_variable = 0
    entry.as_module = scope
    self.add_imported_module(scope)
    return entry