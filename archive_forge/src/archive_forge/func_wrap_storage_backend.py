import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
@property
def wrap_storage_backend(self: torch.storage._StorageBase) -> bool:
    """Return the internal :class:`torch.UntypedStorage`."""
    return self.device.type == custom_backend_name