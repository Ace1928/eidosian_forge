import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def lower_finalize_func(self, lower):
    """
        Lower the generator's finalizer.
        """
    fnty = llvmlite.ir.FunctionType(llvmlite.ir.VoidType(), [self.context.get_value_type(self.gentype)])
    function = cgutils.get_or_insert_function(lower.module, fnty, self.gendesc.llvm_finalizer_name)
    entry_block = function.append_basic_block('entry')
    builder = IRBuilder(entry_block)
    genptrty = self.context.get_value_type(self.gentype)
    genptr = builder.bitcast(function.args[0], genptrty)
    self.lower_finalize_func_body(builder, genptr)