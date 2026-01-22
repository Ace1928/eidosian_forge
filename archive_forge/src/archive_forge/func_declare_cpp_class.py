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
def declare_cpp_class(self, name, scope, pos, cname=None, base_classes=(), visibility='extern', templates=None):
    if cname is None:
        if self.in_cinclude or visibility != 'private':
            cname = name
        else:
            cname = self.mangle(Naming.type_prefix, name)
    base_classes = list(base_classes)
    entry = self.lookup_here(name)
    if not entry:
        type = PyrexTypes.CppClassType(name, scope, cname, base_classes, templates=templates)
        entry = self.declare_type(name, type, pos, cname, visibility=visibility, defining=scope is not None)
        self.sue_entries.append(entry)
    else:
        if not (entry.is_type and entry.type.is_cpp_class):
            error(pos, "'%s' redeclared " % name)
            entry.already_declared_here()
            return None
        elif scope and entry.type.scope:
            warning(pos, "'%s' already defined  (ignoring second definition)" % name, 0)
        elif scope:
            entry.type.scope = scope
            self.type_entries.append(entry)
        if base_classes:
            if entry.type.base_classes and entry.type.base_classes != base_classes:
                error(pos, 'Base type does not match previous declaration')
                entry.already_declared_here()
            else:
                entry.type.base_classes = base_classes
        if templates or entry.type.templates:
            if templates != entry.type.templates:
                error(pos, 'Template parameters do not match previous declaration')
                entry.already_declared_here()

    def declare_inherited_attributes(entry, base_classes):
        for base_class in base_classes:
            if base_class is PyrexTypes.error_type:
                continue
            if base_class.scope is None:
                error(pos, 'Cannot inherit from incomplete type')
            else:
                declare_inherited_attributes(entry, base_class.base_classes)
                entry.type.scope.declare_inherited_cpp_attributes(base_class)
    if scope:
        declare_inherited_attributes(entry, base_classes)
        scope.declare_var(name='this', cname='this', type=PyrexTypes.CPtrType(entry.type), pos=entry.pos)
    if self.is_cpp_class_scope:
        entry.type.namespace = self.outer_scope.lookup(self.name).type
    return entry