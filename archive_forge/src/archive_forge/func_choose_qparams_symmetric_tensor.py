import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'choose_qparams_symmetric.tensor', 'CompositeExplicitAutograd')
def choose_qparams_symmetric_tensor(input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    assert dtype in _DTYPE_TO_QVALUE_BOUNDS, f'Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}'
    validate_qmin_qmax(qmin, qmax)
    min_val, max_val = torch.aminmax(input)
    return determine_qparams(min_val, max_val, qmin, qmax, dtype, torch.Tensor([eps]), has_customized_qrange=False, qscheme=torch.per_tensor_symmetric)