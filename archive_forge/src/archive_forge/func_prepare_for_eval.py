from functools import partial
import torch
import torch.nn as nn
from bitsandbytes.triton.dequantize_rowwise import dequantize_rowwise
from bitsandbytes.triton.int8_matmul_mixed_dequantize import (
from bitsandbytes.triton.int8_matmul_rowwise_dequantize import (
from bitsandbytes.triton.quantize_columnwise_and_transpose import (
from bitsandbytes.triton.quantize_global import (
from bitsandbytes.triton.quantize_rowwise import quantize_rowwise
from bitsandbytes.triton.triton_utils import is_triton_available
def prepare_for_eval(self):
    print('=> preparing for eval.')
    if self.vector_wise_quantization:
        W_int8, state_W = quantize_rowwise(self.weight)
    else:
        W_int8, state_W = quantize_global(self.weight)
    self.register_buffer('W_int8', W_int8)
    self.register_buffer('state_W', state_W)
    del self.weight