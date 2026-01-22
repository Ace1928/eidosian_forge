import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
def dsd_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, width, out=None):
    if a.stride(2) != 1 and a.stride(3) != 1:
        a = a.contiguous()
    if b.stride(2) != 1 and b.stride(3) != 1:
        b = b.contiguous()
    AS1 = block * spdims[2 if trans_a else 1]
    BS0 = b.size(0)
    BS1 = b.size(1)
    BS3 = b.size(2 if trans_b else 3)
    dtype = a.dtype
    CS0 = BS0
    CS1 = BS1
    CS2 = BS3 if trans_c else AS1
    CS3 = AS1 if trans_c else BS3
    if out is None:
        c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
    else:
        assert out.shape == (CS0, CS1, CS2, CS3)
        c = out
    TILE_N = 128
    grid = lambda meta: [cdiv(BS3, meta['TILE_N']), width, BS0]
    _dsd_kernel[grid](a, b, c, a.stride(0), a.stride(1), a.stride(3 if trans_a else 2), a.stride(2 if trans_a else 3), b.stride(0), b.stride(1), b.stride(3 if trans_b else 2), b.stride(2 if trans_b else 3), c.stride(0), c.stride(1), c.stride(3 if trans_c else 2), c.stride(2 if trans_c else 3), BS3, AS1, lut, TILE_M=block, TILE_N=TILE_N, TILE_K=min(block, 32), BLOCK=block, num_stages=4, num_warps=4, GROUP_SIZE_M=4)
    return c