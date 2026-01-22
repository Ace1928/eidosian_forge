import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'dequantize_per_tensor', 'CompositeExplicitAutograd')
def dequantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    """ Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

    Returns:
       dequantized float32 Tensor
    """
    assert input.dtype == dtype, f'Expecting input to have dtype: {dtype}, but got {input.dtype}'
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return (input.to(torch.float32) - zero_point) * scale
    else:
        raise ValueError(f'Unsupported dtype in dequantize_per_tensor: {dtype}')