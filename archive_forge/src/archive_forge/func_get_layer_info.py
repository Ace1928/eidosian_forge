import logging
from typing import List, Tuple
import torch
import torch.nn as nn
def get_layer_info(self) -> List[LayerInfo]:
    """
        Returns a list of LayerInfo instances of the model.
        """
    return self.layer_info