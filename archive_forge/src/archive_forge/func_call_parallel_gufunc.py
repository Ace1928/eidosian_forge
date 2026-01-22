import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def call_parallel_gufunc(lowerer, cres, gu_signature, outer_sig, expr_args, expr_arg_types, loop_ranges, redvars, reddict, redarrdict, init_block, index_var_typ, races, exp_name_to_tuple_var):
    """
    Adds the call to the gufunc function from the main function.
    """
    context = lowerer.context
    builder = lowerer.builder
    from numba.np.ufunc.parallel import build_gufunc_wrapper, _launch_threads
    if config.DEBUG_ARRAY_OPT:
        print('make_parallel_loop')
        print('outer_sig = ', outer_sig.args, outer_sig.return_type, outer_sig.recvr, outer_sig.pysig)
        print('loop_ranges = ', loop_ranges)
        print('expr_args', expr_args)
        print('expr_arg_types', expr_arg_types)
        print('gu_signature', gu_signature)
    args, return_type = sigutils.normalize_signature(outer_sig)
    llvm_func = cres.library.get_function(cres.fndesc.llvm_func_name)
    sin, sout = gu_signature
    _launch_threads()
    info = build_gufunc_wrapper(llvm_func, cres, sin, sout, cache=False, is_parfors=True)
    wrapper_name = info.name
    cres.library._ensure_finalized()
    if config.DEBUG_ARRAY_OPT:
        print('parallel function = ', wrapper_name, cres)

    def load_range(v):
        if isinstance(v, ir.Var):
            return lowerer.loadvar(v.name)
        else:
            return context.get_constant(types.uintp, v)
    num_dim = len(loop_ranges)
    for i in range(num_dim):
        start, stop, step = loop_ranges[i]
        start = load_range(start)
        stop = load_range(stop)
        assert step == 1
        step = load_range(step)
        loop_ranges[i] = (start, stop, step)
        if config.DEBUG_ARRAY_OPT:
            print('call_parallel_gufunc loop_ranges[{}] = '.format(i), start, stop, step)
            cgutils.printf(builder, 'loop range[{}]: %d %d (%d)\n'.format(i), start, stop, step)
    byte_t = llvmlite.ir.IntType(8)
    byte_ptr_t = llvmlite.ir.PointerType(byte_t)
    byte_ptr_ptr_t = llvmlite.ir.PointerType(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    uintp_t = context.get_value_type(types.uintp)
    intp_ptr_t = llvmlite.ir.PointerType(intp_t)
    intp_ptr_ptr_t = llvmlite.ir.PointerType(intp_ptr_t)
    uintp_ptr_t = llvmlite.ir.PointerType(uintp_t)
    uintp_ptr_ptr_t = llvmlite.ir.PointerType(uintp_ptr_t)
    zero = context.get_constant(types.uintp, 0)
    one = context.get_constant(types.uintp, 1)
    one_type = one.type
    sizeof_intp = context.get_abi_sizeof(intp_t)
    expr_args.pop(0)
    sched_sig = sin.pop(0)
    if config.DEBUG_ARRAY_OPT:
        print('Parfor has potentially negative start', index_var_typ.signed)
    if index_var_typ.signed:
        sched_type = intp_t
        sched_ptr_type = intp_ptr_t
        sched_ptr_ptr_type = intp_ptr_ptr_t
    else:
        sched_type = uintp_t
        sched_ptr_type = uintp_ptr_t
        sched_ptr_ptr_type = uintp_ptr_ptr_t
    dim_starts = cgutils.alloca_once(builder, sched_type, size=context.get_constant(types.uintp, num_dim), name='dim_starts')
    dim_stops = cgutils.alloca_once(builder, sched_type, size=context.get_constant(types.uintp, num_dim), name='dim_stops')
    for i in range(num_dim):
        start, stop, step = loop_ranges[i]
        if start.type != one_type:
            start = builder.sext(start, one_type)
        if stop.type != one_type:
            stop = builder.sext(stop, one_type)
        if step.type != one_type:
            step = builder.sext(step, one_type)
        stop = builder.sub(stop, one)
        builder.store(start, builder.gep(dim_starts, [context.get_constant(types.uintp, i)]))
        builder.store(stop, builder.gep(dim_stops, [context.get_constant(types.uintp, i)]))
    get_chunksize = cgutils.get_or_insert_function(builder.module, llvmlite.ir.FunctionType(uintp_t, []), name='get_parallel_chunksize')
    set_chunksize = cgutils.get_or_insert_function(builder.module, llvmlite.ir.FunctionType(llvmlite.ir.VoidType(), [uintp_t]), name='set_parallel_chunksize')
    get_num_threads = cgutils.get_or_insert_function(builder.module, llvmlite.ir.FunctionType(llvmlite.ir.IntType(types.intp.bitwidth), []), 'get_num_threads')
    num_threads = builder.call(get_num_threads, [])
    current_chunksize = builder.call(get_chunksize, [])
    with cgutils.if_unlikely(builder, builder.icmp_signed('<=', num_threads, num_threads.type(0))):
        cgutils.printf(builder, 'num_threads: %d\n', num_threads)
        context.call_conv.return_user_exc(builder, RuntimeError, ('Invalid number of threads. This likely indicates a bug in Numba.',))
    get_sched_size_fnty = llvmlite.ir.FunctionType(uintp_t, [uintp_t, uintp_t, intp_ptr_t, intp_ptr_t])
    get_sched_size = cgutils.get_or_insert_function(builder.module, get_sched_size_fnty, name='get_sched_size')
    num_divisions = builder.call(get_sched_size, [num_threads, context.get_constant(types.uintp, num_dim), dim_starts, dim_stops])
    builder.call(set_chunksize, [zero])
    multiplier = context.get_constant(types.uintp, num_dim * 2)
    sched_size = builder.mul(num_divisions, multiplier)
    alloc_sched_fnty = llvmlite.ir.FunctionType(sched_ptr_type, [uintp_t])
    alloc_sched_func = cgutils.get_or_insert_function(builder.module, alloc_sched_fnty, name='allocate_sched')
    alloc_space = builder.call(alloc_sched_func, [sched_size])
    sched = cgutils.alloca_once(builder, sched_ptr_type)
    builder.store(alloc_space, sched)
    debug_flag = 1 if config.DEBUG_ARRAY_OPT else 0
    scheduling_fnty = llvmlite.ir.FunctionType(intp_ptr_t, [uintp_t, intp_ptr_t, intp_ptr_t, uintp_t, sched_ptr_type, intp_t])
    if index_var_typ.signed:
        do_scheduling = cgutils.get_or_insert_function(builder.module, scheduling_fnty, name='do_scheduling_signed')
    else:
        do_scheduling = cgutils.get_or_insert_function(builder.module, scheduling_fnty, name='do_scheduling_unsigned')
    builder.call(do_scheduling, [context.get_constant(types.uintp, num_dim), dim_starts, dim_stops, num_divisions, builder.load(sched), context.get_constant(types.intp, debug_flag)])
    redarrs = [lowerer.loadvar(redarrdict[x].name) for x in redvars]
    nredvars = len(redvars)
    ninouts = len(expr_args) - nredvars

    def load_potential_tuple_var(x):
        """Given a variable name, if that variable is not a new name
           introduced as the extracted part of a tuple then just return
           the variable loaded from its name.  However, if the variable
           does represent part of a tuple, as recognized by the name of
           the variable being present in the exp_name_to_tuple_var dict,
           then we load the original tuple var instead that we get from
           the dict and then extract the corresponding element of the
           tuple, also stored and returned to use in the dict (i.e., offset).
        """
        if x in exp_name_to_tuple_var:
            orig_tup, offset = exp_name_to_tuple_var[x]
            tup_var = lowerer.loadvar(orig_tup)
            res = builder.extract_value(tup_var, offset)
            return res
        else:
            return lowerer.loadvar(x)
    all_args = [load_potential_tuple_var(x) for x in expr_args[:ninouts]] + redarrs
    num_args = len(all_args)
    num_inps = len(sin) + 1
    args = cgutils.alloca_once(builder, byte_ptr_t, size=context.get_constant(types.intp, 1 + num_args), name='pargs')
    array_strides = []
    builder.store(builder.bitcast(builder.load(sched), byte_ptr_t), args)
    array_strides.append(context.get_constant(types.intp, sizeof_intp))
    rv_to_arg_dict = {}
    for i in range(num_args):
        arg = all_args[i]
        var = expr_args[i]
        aty = expr_arg_types[i]
        dst = builder.gep(args, [context.get_constant(types.intp, i + 1)])
        if i >= ninouts:
            ary = context.make_array(aty)(context, builder, arg)
            strides = cgutils.unpack_tuple(builder, ary.strides, aty.ndim)
            for j in range(len(strides)):
                array_strides.append(strides[j])
            builder.store(builder.bitcast(ary.data, byte_ptr_t), dst)
        elif isinstance(aty, types.ArrayCompatible):
            if var in races:
                typ = context.get_data_type(aty.dtype) if aty.dtype != types.boolean else llvmlite.ir.IntType(1)
                rv_arg = cgutils.alloca_once(builder, typ)
                builder.store(arg, rv_arg)
                builder.store(builder.bitcast(rv_arg, byte_ptr_t), dst)
                rv_to_arg_dict[var] = (arg, rv_arg)
                array_strides.append(context.get_constant(types.intp, context.get_abi_sizeof(typ)))
            else:
                ary = context.make_array(aty)(context, builder, arg)
                strides = cgutils.unpack_tuple(builder, ary.strides, aty.ndim)
                for j in range(len(strides)):
                    array_strides.append(strides[j])
                builder.store(builder.bitcast(ary.data, byte_ptr_t), dst)
        else:
            if i < num_inps:
                if isinstance(aty, types.Optional):
                    unpacked_aty = aty.type
                    arg = context.cast(builder, arg, aty, unpacked_aty)
                else:
                    unpacked_aty = aty
                typ = context.get_data_type(unpacked_aty) if not isinstance(unpacked_aty, types.Boolean) else llvmlite.ir.IntType(1)
                ptr = cgutils.alloca_once(builder, typ)
                builder.store(arg, ptr)
            else:
                typ = context.get_data_type(aty) if not isinstance(aty, types.Boolean) else llvmlite.ir.IntType(1)
                ptr = cgutils.alloca_once(builder, typ)
            builder.store(builder.bitcast(ptr, byte_ptr_t), dst)
    sig_dim_dict = {}
    occurrences = []
    occurrences = [sched_sig[0]]
    sig_dim_dict[sched_sig[0]] = context.get_constant(types.intp, 2 * num_dim)
    assert len(expr_args) == len(all_args)
    assert len(expr_args) == len(expr_arg_types)
    assert len(expr_args) == len(sin + sout)
    assert len(expr_args) == len(outer_sig.args[1:])
    for var, arg, aty, gu_sig in zip(expr_args, all_args, expr_arg_types, sin + sout):
        if isinstance(aty, types.npytypes.Array):
            i = aty.ndim - len(gu_sig)
        else:
            i = 0
        if config.DEBUG_ARRAY_OPT:
            print('var =', var, 'gu_sig =', gu_sig, 'type =', aty, 'i =', i)
        for dim_sym in gu_sig:
            if config.DEBUG_ARRAY_OPT:
                print('var = ', var, ' type = ', aty)
            if var in races:
                sig_dim_dict[dim_sym] = context.get_constant(types.intp, 1)
            else:
                ary = context.make_array(aty)(context, builder, arg)
                shapes = cgutils.unpack_tuple(builder, ary.shape, aty.ndim)
                sig_dim_dict[dim_sym] = shapes[i]
            if not dim_sym in occurrences:
                if config.DEBUG_ARRAY_OPT:
                    print('dim_sym = ', dim_sym, ', i = ', i)
                    cgutils.printf(builder, dim_sym + ' = %d\n', sig_dim_dict[dim_sym])
                occurrences.append(dim_sym)
            i = i + 1
    nshapes = len(sig_dim_dict) + 1
    shapes = cgutils.alloca_once(builder, intp_t, size=nshapes, name='pshape')
    builder.store(num_divisions, shapes)
    i = 1
    for dim_sym in occurrences:
        if config.DEBUG_ARRAY_OPT:
            cgutils.printf(builder, dim_sym + ' = %d\n', sig_dim_dict[dim_sym])
        builder.store(sig_dim_dict[dim_sym], builder.gep(shapes, [context.get_constant(types.intp, i)]))
        i = i + 1
    num_steps = num_args + 1 + len(array_strides)
    steps = cgutils.alloca_once(builder, intp_t, size=context.get_constant(types.intp, num_steps), name='psteps')
    builder.store(context.get_constant(types.intp, 2 * num_dim * sizeof_intp), steps)
    for i in range(num_args):
        stepsize = zero
        dst = builder.gep(steps, [context.get_constant(types.intp, 1 + i)])
        builder.store(stepsize, dst)
    for j in range(len(array_strides)):
        dst = builder.gep(steps, [context.get_constant(types.intp, 1 + num_args + j)])
        builder.store(array_strides[j], dst)
    data = cgutils.get_null_value(byte_ptr_t)
    fnty = llvmlite.ir.FunctionType(llvmlite.ir.VoidType(), [byte_ptr_ptr_t, intp_ptr_t, intp_ptr_t, byte_ptr_t])
    fn = cgutils.get_or_insert_function(builder.module, fnty, wrapper_name)
    context.active_code_library.add_linking_library(info.library)
    if config.DEBUG_ARRAY_OPT:
        cgutils.printf(builder, 'before calling kernel %p\n', fn)
    builder.call(fn, [args, shapes, steps, data])
    if config.DEBUG_ARRAY_OPT:
        cgutils.printf(builder, 'after calling kernel %p\n', fn)
    builder.call(set_chunksize, [current_chunksize])
    dealloc_sched_fnty = llvmlite.ir.FunctionType(llvmlite.ir.VoidType(), [sched_ptr_type])
    dealloc_sched_func = cgutils.get_or_insert_function(builder.module, dealloc_sched_fnty, name='deallocate_sched')
    builder.call(dealloc_sched_func, [builder.load(sched)])
    for k, v in rv_to_arg_dict.items():
        arg, rv_arg = v
        only_elem_ptr = builder.gep(rv_arg, [context.get_constant(types.intp, 0)])
        builder.store(builder.load(only_elem_ptr), lowerer.getvar(k))
    context.active_code_library.add_linking_library(cres.library)