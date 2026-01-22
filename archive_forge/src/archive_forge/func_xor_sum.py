from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@core.builtin
@core._add_reduction_docstr('xor sum')
def xor_sum(input, axis=None, _builder=None, _generator=None):
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError('xor_sum only supported for integers')
    input = core._promote_reduction_input(input, _builder=_builder)
    return core.reduce(input, axis, _xor_combine, _builder=_builder, _generator=_generator)