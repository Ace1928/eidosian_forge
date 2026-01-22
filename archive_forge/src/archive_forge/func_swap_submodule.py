from typing import Dict, Iterable, List, Tuple
import torch
def swap_submodule(self, path: str, value: 'torch.nn.Module') -> 'torch.nn.Module':
    """
        Swap the submodule specified by the given ``path`` to ``value``.

        For example, to swap the attribute mod.layer1.conv1 use
        ``accessor.swap_submodule("layer1.conv1", conv2)``.
        """
    prefix, _, attr = path.rpartition('.')
    return swap_submodule(self.get_submodule(prefix), attr, value)