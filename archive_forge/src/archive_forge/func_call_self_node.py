from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def call_self_node(self, omit_optional_args=0, is_module_scope=0):
    from . import ExprNodes
    args = self.type.args
    if omit_optional_args:
        args = args[:len(args) - self.type.optional_arg_count]
    arg_names = [arg.name for arg in args]
    if is_module_scope:
        cfunc = ExprNodes.NameNode(self.pos, name=self.entry.name)
        call_arg_names = arg_names
        skip_dispatch = Options.lookup_module_cpdef
    elif self.type.is_static_method:
        class_entry = self.entry.scope.parent_type.entry
        class_node = ExprNodes.NameNode(self.pos, name=class_entry.name)
        class_node.entry = class_entry
        cfunc = ExprNodes.AttributeNode(self.pos, obj=class_node, attribute=self.entry.name)
        skip_dispatch = True
    else:
        type_entry = self.type.args[0].type.entry
        type_arg = ExprNodes.NameNode(self.pos, name=type_entry.name)
        type_arg.entry = type_entry
        cfunc = ExprNodes.AttributeNode(self.pos, obj=type_arg, attribute=self.entry.name)
    skip_dispatch = not is_module_scope or Options.lookup_module_cpdef
    c_call = ExprNodes.SimpleCallNode(self.pos, function=cfunc, args=[ExprNodes.NameNode(self.pos, name=n) for n in arg_names], wrapper_call=skip_dispatch)
    return ReturnStatNode(pos=self.pos, return_type=PyrexTypes.py_object_type, value=c_call)