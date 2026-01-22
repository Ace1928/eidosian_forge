from typing import Dict, Iterable, List, Tuple
import torch
def swap_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor], allow_missing: bool=False) -> List[torch.Tensor]:
    """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
    if not isinstance(names, (list, tuple)):
        names = list(names)
    if not isinstance(values, (list, tuple)):
        values = list(values)
    assert len(names) == len(values), 'names and values must have the same length'
    return [self.swap_tensor(name, value, allow_missing=allow_missing) for name, value in zip(names, values)]