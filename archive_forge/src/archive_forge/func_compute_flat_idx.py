import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
@intrinsic
def compute_flat_idx(typingctx, strides, itemsize, idx, axis):
    sig = types.intp(strides, itemsize, idx, axis)
    len_idx = len(idx)

    def gen_block(builder, block_pos, block_name, bb_end, args):
        strides, _, idx, _ = args
        bb = builder.append_basic_block(name=block_name)
        with builder.goto_block(bb):
            zero = ir.IntType(64)(0)
            flat_idx = zero
            if block_pos == 0:
                for i in range(1, len_idx):
                    stride = builder.extract_value(strides, i - 1)
                    idx_i = builder.extract_value(idx, i)
                    m = builder.mul(stride, idx_i)
                    flat_idx = builder.add(flat_idx, m)
            elif 0 < block_pos < len_idx - 1:
                for i in range(0, block_pos):
                    stride = builder.extract_value(strides, i)
                    idx_i = builder.extract_value(idx, i)
                    m = builder.mul(stride, idx_i)
                    flat_idx = builder.add(flat_idx, m)
                for i in range(block_pos + 1, len_idx):
                    stride = builder.extract_value(strides, i - 1)
                    idx_i = builder.extract_value(idx, i)
                    m = builder.mul(stride, idx_i)
                    flat_idx = builder.add(flat_idx, m)
            else:
                for i in range(0, len_idx - 1):
                    stride = builder.extract_value(strides, i)
                    idx_i = builder.extract_value(idx, i)
                    m = builder.mul(stride, idx_i)
                    flat_idx = builder.add(flat_idx, m)
            builder.branch(bb_end)
        return (bb, flat_idx)

    def codegen(context, builder, sig, args):
        strides, itemsize, idx, axis = args
        bb = builder.basic_block
        switch_end = builder.append_basic_block(name='axis_end')
        l = []
        for i in range(len_idx):
            block, flat_idx = gen_block(builder, i, f'axis_{i}', switch_end, args)
            l.append((block, flat_idx))
        with builder.goto_block(bb):
            switch = builder.switch(axis, l[-1][0])
            for i in range(len_idx):
                switch.add_case(i, l[i][0])
        builder.position_at_end(switch_end)
        phi = builder.phi(l[0][1].type)
        for block, value in l:
            phi.add_incoming(value, block)
        return builder.sdiv(phi, itemsize)
    return (sig, codegen)