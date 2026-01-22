import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
def get_current_mesh(self) -> 'DeviceMesh':
    if len(self.mesh_stack) == 0:
        raise RuntimeError('No device mesh is currently active!')
    return self.mesh_stack[-1]