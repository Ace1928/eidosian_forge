import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
@property
def wrap_typed_storage_backend(self: torch.storage.TypedStorage) -> bool:
    torch.storage._warn_typed_storage_removal()
    return self._untyped_storage.device.type == custom_backend_name