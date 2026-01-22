import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'choose_qparams_symmetric.tensor', 'Meta')
def choose_qparams_symmetric_tensor_meta(input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device))