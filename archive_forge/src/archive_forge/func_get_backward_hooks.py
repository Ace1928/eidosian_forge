import logging
from typing import List, Tuple
import torch
import torch.nn as nn
def get_backward_hooks(self) -> List:
    """
        Returns a list of tuples. Each tuple contains the layer name and the
        hook attached to it.
        """
    layer_name_and_hooks = list()
    for name, layer in self._model.named_modules():
        if name != '':
            layer_name_and_hooks.append((name, layer._get_backward_hooks()))
    return layer_name_and_hooks