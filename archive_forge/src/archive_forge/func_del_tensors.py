from typing import Dict, Iterable, List, Tuple
import torch
def del_tensors(self, names: Iterable[str]) -> None:
    """
        Delete the attributes specified by the given paths.

        For example, to delete the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.del_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
    for name in names:
        self.del_tensor(name)