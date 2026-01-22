from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
def replace_callee(self, newfunc):
    if newfunc.function_type != self.callee.function_type:
        raise TypeError('New function has incompatible type')
    self.callee = newfunc