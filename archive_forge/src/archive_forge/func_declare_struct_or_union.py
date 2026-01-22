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
def declare_struct_or_union(self, name, kind, scope, typedef_flag, pos, cname=None, visibility='private', api=0, packed=False):
    if not cname:
        if self.in_cinclude or (visibility == 'public' or api):
            cname = name
        else:
            cname = self.mangle(Naming.type_prefix, name)
    entry = self.lookup_here(name)
    if not entry:
        in_cpp = self.is_cpp()
        type = PyrexTypes.CStructOrUnionType(name, kind, scope, typedef_flag, cname, packed, in_cpp=in_cpp)
        entry = self.declare_type(name, type, pos, cname, visibility=visibility, api=api, defining=scope is not None)
        self.sue_entries.append(entry)
        type.entry = entry
    elif not (entry.is_type and entry.type.is_struct_or_union and (entry.type.kind == kind)):
        warning(pos, "'%s' redeclared  " % name, 0)
    elif scope and entry.type.scope:
        warning(pos, "'%s' already defined  (ignoring second definition)" % name, 0)
    else:
        self.check_previous_typedef_flag(entry, typedef_flag, pos)
        self.check_previous_visibility(entry, visibility, pos)
        if scope:
            entry.type.scope = scope
            self.type_entries.append(entry)
    if self.is_cpp_class_scope:
        entry.type.namespace = self.outer_scope.lookup(self.name).type
    return entry