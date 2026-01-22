from typing import Dict, Iterable, List, Tuple
import torch
def set_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor]) -> None:
    """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
    if not isinstance(names, (list, tuple)):
        names = list(names)
    if not isinstance(values, (list, tuple)):
        values = list(values)
    assert len(names) == len(values), 'names and values must have the same length'
    for name, value in zip(names, values):
        self.set_tensor(name, value)