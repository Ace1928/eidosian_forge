from __future__ import absolute_import
from .Symtab import ModuleScope
from .PyrexTypes import *
from .UtilityCode import CythonUtilityCode
from .Errors import error
from .Scanning import StringSourceDescriptor
from . import MemoryView
from .StringEncoding import EncodedString
def lookup_qualified_name(self, qname):
    name_path = qname.split(u'.')
    scope = self
    while len(name_path) > 1:
        scope = scope.lookup_here(name_path[0])
        if scope:
            scope = scope.as_module
        del name_path[0]
        if scope is None:
            return None
    else:
        return scope.lookup_here(name_path[0])