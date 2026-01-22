import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def wrap_storage_to(self, device=None, non_blocking=False):
    """Return a copy of this object in custom device memory.

        If this object is already in device memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination device id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        """
    device_idx = _normalization_device(custom_backend_name, device)
    if getattr(self, f'is_{custom_backend_name}'):
        if self.get_device() == device_idx:
            return self
    if self.is_sparse:
        raise RuntimeError(f'Can not support a sparse storage move to {custom_backend_name} backend')
    untyped_storage = torch.UntypedStorage(self.size(), device=torch.device(f'{custom_backend_name}:{device_idx}'))
    untyped_storage.copy_(self, non_blocking)
    return untyped_storage