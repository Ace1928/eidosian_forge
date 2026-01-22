import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
@property
def llvm_finalizer_name(self):
    """
        The LLVM name of the generator's finalizer function
        (if <generator type>.has_finalizer is true).
        """
    return 'finalize_' + self.mangled_name