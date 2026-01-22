import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
def dds_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, width, out=None):
    return dsd_matmul(b, a, not trans_b, not trans_a, not trans_c, spdims, block, lut, width, out=out)