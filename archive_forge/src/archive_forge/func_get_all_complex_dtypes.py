from typing import List
import torch
def get_all_complex_dtypes(include_complex32=False) -> List[torch.dtype]:
    return [torch.complex32, torch.complex64, torch.complex128] if include_complex32 else [torch.complex64, torch.complex128]