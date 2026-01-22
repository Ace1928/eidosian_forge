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
def check_c_class(self, entry):
    type = entry.type
    name = entry.name
    visibility = entry.visibility
    if not type.scope:
        error(entry.pos, "C class '%s' is declared but not defined" % name)
    if visibility != 'extern' and (not type.typeobj_cname):
        type.typeobj_cname = self.mangle(Naming.typeobj_prefix, name)
    if type.scope:
        for method_entry in type.scope.cfunc_entries:
            if not method_entry.is_inherited and (not method_entry.func_cname):
                error(method_entry.pos, "C method '%s' is declared but not defined" % method_entry.name)
    if type.vtabslot_cname:
        type.vtable_cname = self.mangle(Naming.vtable_prefix, entry.name)