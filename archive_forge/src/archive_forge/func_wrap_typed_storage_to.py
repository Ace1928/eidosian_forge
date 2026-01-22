import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def wrap_typed_storage_to(self: torch.storage.TypedStorage, device=None, non_blocking=False, **kwargs) -> torch.storage.TypedStorage:
    torch.storage._warn_typed_storage_removal()
    if unsupported_dtype and self.dtype in unsupported_dtype:
        raise RuntimeError(f'Cannot create {custom_backend_name} storage as {self.dtype} dtype is not supported by this backend')
    custom_backend_storage: torch.UntypedStorage = getattr(self._untyped_storage, custom_backend_name)(device, non_blocking, **kwargs)
    return self._new_wrapped_storage(custom_backend_storage)