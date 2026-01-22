import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def get_item_pointer2(context, builder, data, shape, strides, layout, inds, wraparound=False, boundscheck=False):
    if wraparound:
        indices = []
        for ind, dimlen in zip(inds, shape):
            negative = builder.icmp_signed('<', ind, ind.type(0))
            wrapped = builder.add(dimlen, ind)
            selected = builder.select(negative, wrapped, ind)
            indices.append(selected)
    else:
        indices = inds
    if boundscheck:
        for axis, (ind, dimlen) in enumerate(zip(indices, shape)):
            do_boundscheck(context, builder, ind, dimlen, axis)
    if not indices:
        return builder.gep(data, [int32_t(0)])
    intp = indices[0].type
    if layout in 'CF':
        steps = []
        if layout == 'C':
            for i in range(len(shape)):
                last = intp(1)
                for j in shape[i + 1:]:
                    last = builder.mul(last, j)
                steps.append(last)
        elif layout == 'F':
            for i in range(len(shape)):
                last = intp(1)
                for j in shape[:i]:
                    last = builder.mul(last, j)
                steps.append(last)
        else:
            raise Exception('unreachable')
        loc = intp(0)
        for i, s in zip(indices, steps):
            tmp = builder.mul(i, s)
            loc = builder.add(loc, tmp)
        ptr = builder.gep(data, [loc])
        return ptr
    else:
        dimoffs = [builder.mul(s, i) for s, i in zip(strides, indices)]
        offset = functools.reduce(builder.add, dimoffs)
        return pointer_add(builder, data, offset)