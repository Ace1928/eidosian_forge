from typing import Dict, Iterable, List, Union, cast
from ..compat import has_torch_amp, torch
from ..util import is_torch_array
def unscale(self, tensors):
    """Unscale the given tensors. Returns True if any of the gradients were infinite."""
    if not self._enabled:
        return False
    inv_scale = self._scale.double().reciprocal().float()
    tensors_per_device = self._tensors_per_device(tensors)
    for device, device_tensors in tensors_per_device.items():
        found_inf_device = torch.full((1,), 0.0, device=device)
        inv_scale_device = inv_scale.to(device=device)
        torch._amp_foreach_non_finite_check_and_unscale_(device_tensors, found_inf_device, inv_scale_device)
        if bool(found_inf_device != 0):
            self._found_inf = True
    return self._found_inf