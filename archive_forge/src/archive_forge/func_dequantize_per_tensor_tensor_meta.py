import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'dequantize_per_tensor.tensor', 'Meta')
def dequantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    assert input.dtype == dtype, f'Expecting input to have dtype: {dtype}'
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return torch.empty_like(input, dtype=torch.float32)
    else:
        raise ValueError(f'Unsupported dtype in dequantize_per_tensor: {dtype}')