from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def ravel(x):
    """
    Returns a contiguous flattened view of :code:`x`.

    :param x: the input tensor
    :type x: Block
    """
    return core.view(x, [x.numel])