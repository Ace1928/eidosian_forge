from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
def build_func_type(self, sig=None, self_arg=None):
    if sig is None:
        sig = Signature(self.args, self.ret_type, nogil=self.nogil)
        sig.exception_check = False
    func_type = sig.function_type(self_arg)
    if self.is_strict_signature:
        func_type.is_strict_signature = True
    if self.builtin_return_type:
        func_type.return_type = builtin_types[self.builtin_return_type]
    return func_type