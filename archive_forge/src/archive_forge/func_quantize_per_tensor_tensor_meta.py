import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'quantize_per_tensor.tensor', 'Meta')
def quantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    return torch.empty_like(input, dtype=dtype)