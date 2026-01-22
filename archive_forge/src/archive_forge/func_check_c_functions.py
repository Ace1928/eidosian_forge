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
def check_c_functions(self):
    for name, entry in self.entries.items():
        if entry.is_cfunction:
            if entry.defined_in_pxd and entry.scope is self and (entry.visibility != 'extern') and (not entry.in_cinclude) and (not entry.is_implemented):
                error(entry.pos, "Non-extern C function '%s' declared but not defined" % name)