import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
def sdd_lut(layout, block, device):
    lut = layout.nonzero(as_tuple=False).to(device).int()
    lut = lut.contiguous()
    return (lut, None)