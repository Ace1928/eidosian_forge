import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@staticmethod
def fake_8bit_storage_stoachstic(w):
    rand = torch.rand(1024, device=w.device)
    absmax, C = bnb.functional.quantize_blockwise(w.data, rand=rand)
    out = bnb.functional.dequantize_blockwise(absmax, C)
    out = out.half()
    w.copy_(out)
    return out