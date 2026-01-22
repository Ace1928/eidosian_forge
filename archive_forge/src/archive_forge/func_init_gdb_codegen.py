import os
import sys
from llvmlite import ir
from numba.core import types, utils, config, cgutils, errors
from numba import gdb, gdb_init, gdb_breakpoint
from numba.core.extending import overload, intrinsic
def init_gdb_codegen(cgctx, builder, signature, args, const_args, do_break=False):
    int8_t = ir.IntType(8)
    int32_t = ir.IntType(32)
    intp_t = ir.IntType(utils.MACHINE_BITS)
    char_ptr = ir.PointerType(ir.IntType(8))
    zero_i32t = int32_t(0)
    mod = builder.module
    pid = cgutils.alloca_once(builder, int32_t, size=1)
    pidstr = cgutils.alloca_once(builder, int8_t, size=12)
    intfmt = cgctx.insert_const_string(mod, '%d')
    gdb_str = cgctx.insert_const_string(mod, config.GDB_BINARY)
    attach_str = cgctx.insert_const_string(mod, 'attach')
    new_args = []
    new_args.extend(['-x', os.path.join(_path, 'cmdlang.gdb')])
    new_args.extend(['-ex', 'c'])
    if any([not isinstance(x, types.StringLiteral) for x in const_args]):
        raise errors.RequireLiteralValue(const_args)
    new_args.extend([x.literal_value for x in const_args])
    cmdlang = [cgctx.insert_const_string(mod, x) for x in new_args]
    fnty = ir.FunctionType(int32_t, tuple())
    getpid = cgutils.get_or_insert_function(mod, fnty, 'getpid')
    fnty = ir.FunctionType(int32_t, (char_ptr, intp_t, char_ptr), var_arg=True)
    snprintf = cgutils.get_or_insert_function(mod, fnty, 'snprintf')
    fnty = ir.FunctionType(int32_t, tuple())
    fork = cgutils.get_or_insert_function(mod, fnty, 'fork')
    fnty = ir.FunctionType(int32_t, (char_ptr, char_ptr), var_arg=True)
    execl = cgutils.get_or_insert_function(mod, fnty, 'execl')
    fnty = ir.FunctionType(int32_t, (int32_t,))
    sleep = cgutils.get_or_insert_function(mod, fnty, 'sleep')
    fnty = ir.FunctionType(ir.VoidType(), tuple())
    breakpoint = cgutils.get_or_insert_function(mod, fnty, 'numba_gdb_breakpoint')
    parent_pid = builder.call(getpid, tuple())
    builder.store(parent_pid, pid)
    pidstr_ptr = builder.gep(pidstr, [zero_i32t], inbounds=True)
    pid_val = builder.load(pid)
    stat = builder.call(snprintf, (pidstr_ptr, intp_t(12), intfmt, pid_val))
    invalid_write = builder.icmp_signed('>', stat, int32_t(12))
    with builder.if_then(invalid_write, likely=False):
        msg = 'Internal error: `snprintf` buffer would have overflowed.'
        cgctx.call_conv.return_user_exc(builder, RuntimeError, (msg,))
    child_pid = builder.call(fork, tuple())
    fork_failed = builder.icmp_signed('==', child_pid, int32_t(-1))
    with builder.if_then(fork_failed, likely=False):
        msg = 'Internal error: `fork` failed.'
        cgctx.call_conv.return_user_exc(builder, RuntimeError, (msg,))
    is_child = builder.icmp_signed('==', child_pid, zero_i32t)
    with builder.if_else(is_child) as (then, orelse):
        with then:
            nullptr = ir.Constant(char_ptr, None)
            gdb_str_ptr = builder.gep(gdb_str, [zero_i32t], inbounds=True)
            attach_str_ptr = builder.gep(attach_str, [zero_i32t], inbounds=True)
            cgutils.printf(builder, 'Attaching to PID: %s\n', pidstr)
            buf = (gdb_str_ptr, gdb_str_ptr, attach_str_ptr, pidstr_ptr)
            buf = buf + tuple(cmdlang) + (nullptr,)
            builder.call(execl, buf)
        with orelse:
            builder.call(sleep, (int32_t(10),))
            if do_break is True:
                builder.call(breakpoint, tuple())