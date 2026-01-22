from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
def get_known_standard_library_entry(qualified_name):
    name_parts = qualified_name.split('.')
    module_name = EncodedString(name_parts[0])
    rest = name_parts[1:]
    if len(rest) > 1:
        return None
    mod = get_known_standard_library_module_scope(module_name)
    if mod and rest:
        return mod.lookup_here(rest[0])
    return None