import torch
def pack_int4_to_int8(weight):
    assert weight.dim() == 2
    assert weight.shape[1] % 2 == 0
    assert weight.dtype == torch.int8
    return (weight[:, 1::2] & 15) << 4 | weight[:, 0::2] & 15