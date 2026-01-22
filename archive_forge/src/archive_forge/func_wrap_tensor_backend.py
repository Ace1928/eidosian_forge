import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
@property
def wrap_tensor_backend(self: torch.Tensor) -> bool:
    return self.device.type == custom_backend_name