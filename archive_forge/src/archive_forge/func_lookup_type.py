from __future__ import absolute_import
from .Symtab import ModuleScope
from .PyrexTypes import *
from .UtilityCode import CythonUtilityCode
from .Errors import error
from .Scanning import StringSourceDescriptor
from . import MemoryView
from .StringEncoding import EncodedString
def lookup_type(self, name):
    type = parse_basic_type(name)
    if type:
        return type
    return super(CythonScope, self).lookup_type(name)