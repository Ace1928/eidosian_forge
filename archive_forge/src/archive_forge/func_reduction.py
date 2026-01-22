from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def reduction(inputs: Sequence[tl.tensor], axis: int, region_builder_fn, builder: ir.builder) -> Tuple[tl.tensor, ...]:
    if axis is None:
        new_inputs = []
        for i in range(len(inputs)):
            new_shape = [inputs[i].numel.value]
            new_inputs.append(view(inputs[i], new_shape, builder))
        inputs = tuple(new_inputs)
        axis = 0
    shape = inputs[0].type.shape
    ret_shape = [s for i, s in enumerate(shape) if i != axis]
    for t in inputs:
        assert t.type.shape == shape

    def wrap_tensor(x, scalar_ty):
        if ret_shape:
            res_ty = tl.block_type(scalar_ty, ret_shape)
        else:
            res_ty = scalar_ty
        return tl.tensor(x, res_ty)
    reduce_op = builder.create_reduce([t.handle for t in inputs], axis)
    region_builder_fn(reduce_op)
    reduce_op.verify()
    return tuple((wrap_tensor(reduce_op.get_result(i), inputs[i].type.scalar) for i in range(len(inputs))))