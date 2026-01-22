from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return math.min(x, y)