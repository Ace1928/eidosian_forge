import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def lower_init_func(self, lower):
    """
        Lower the generator's initialization function (which will fill up
        the passed-by-reference generator structure).
        """
    lower.setup_function(self.fndesc)
    builder = lower.builder
    lower.context.insert_generator(self.gentype, self.gendesc, [self.library])
    lower.extract_function_arguments()
    lower.pre_lower()
    retty = self.context.get_return_type(self.gentype)
    resume_index = self.context.get_constant(types.int32, 0)
    argsty = retty.elements[1]
    statesty = retty.elements[2]
    lower.debug_print('# low_init_func incref')
    if self.context.enable_nrt:
        for argty, argval in zip(self.fndesc.argtypes, lower.fnargs):
            self.context.nrt.incref(builder, argty, argval)
    argsval = self.arg_packer.as_data(builder, lower.fnargs)
    statesval = Constant(statesty, None)
    gen_struct = cgutils.make_anonymous_struct(builder, [resume_index, argsval, statesval], retty)
    retval = self.box_generator_struct(lower, gen_struct)
    lower.debug_print('# low_init_func before return')
    self.call_conv.return_value(builder, retval)
    lower.post_lower()