import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
def get_parent_mesh(self, device_mesh: 'DeviceMesh') -> Optional['DeviceMesh']:
    return self.child_to_parent_mapping.get(device_mesh, None)