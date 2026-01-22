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
def lookup_target(self, name):
    entry = self.lookup_here(name)
    if not entry:
        entry = self.lookup_here_unmangled(name)
        if entry and entry.is_pyglobal:
            self._emit_class_private_warning(entry.pos, name)
    if not entry:
        entry = self.declare_var(name, py_object_type, None)
    return entry