import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def lower_yield_suspend(self):
    self.lower.debug_print('# generator suspend')
    for state_index, name in zip(self.live_var_indices, self.live_vars):
        state_slot = cgutils.gep_inbounds(self.builder, self.gen_state_ptr, 0, state_index)
        ty = self.gentype.state_types[state_index]
        fetype = self.lower.typeof(name)
        self.lower._alloca_var(name, fetype)
        val = self.lower.loadvar(name)
        if self.context.enable_nrt:
            self.context.nrt.incref(self.builder, ty, val)
        self.context.pack_value(self.builder, ty, val, state_slot)
    indexval = Constant(self.resume_index_ptr.type.pointee, self.inst.index)
    self.builder.store(indexval, self.resume_index_ptr)
    self.lower.debug_print('# generator suspend end')