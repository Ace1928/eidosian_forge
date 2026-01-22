from llvmlite import ir
from numba.core import cgutils, types
from numba.core.imputils import Registry
from numba.cuda import libdevice, libdevicefuncs
def libdevice_implement_multiple_returns(func, retty, prototype_args):
    sig = libdevicefuncs.create_signature(retty, prototype_args)
    nb_retty = sig.return_type

    def core(context, builder, sig, args):
        lmod = builder.module
        fargtys = []
        for arg in prototype_args:
            ty = context.get_value_type(arg.ty)
            if arg.is_ptr:
                ty = ty.as_pointer()
            fargtys.append(ty)
        fretty = context.get_value_type(retty)
        fnty = ir.FunctionType(fretty, fargtys)
        fn = cgutils.get_or_insert_function(lmod, fnty, func)
        actual_args = []
        virtual_args = []
        arg_idx = 0
        for arg in prototype_args:
            if arg.is_ptr:
                tmp_arg = cgutils.alloca_once(builder, context.get_value_type(arg.ty))
                actual_args.append(tmp_arg)
                virtual_args.append(tmp_arg)
            else:
                actual_args.append(args[arg_idx])
                arg_idx += 1
        ret = builder.call(fn, actual_args)
        tuple_args = []
        if retty != types.void:
            tuple_args.append(ret)
        for arg in virtual_args:
            tuple_args.append(builder.load(arg))
        if isinstance(nb_retty, types.UniTuple):
            return cgutils.pack_array(builder, tuple_args)
        else:
            return cgutils.pack_struct(builder, tuple_args)
    key = getattr(libdevice, func[5:])
    lower(key, *sig.args)(core)