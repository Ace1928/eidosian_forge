import contextlib
import functools
from llvmlite.ir import instructions, types, values
def load_reg(self, reg_type, reg_name, name=''):
    """
        Load a register value into an LLVM value.
          Example: v = load_reg(IntType(32), "eax")
        """
    ftype = types.FunctionType(reg_type, [])
    return self.asm(ftype, '', '={%s}' % reg_name, [], False, name)