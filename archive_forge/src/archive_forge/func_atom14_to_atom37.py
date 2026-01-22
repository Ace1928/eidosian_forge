from typing import Dict, Tuple, overload
import torch
import torch.types
from torch import nn
from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather
def atom14_to_atom37(atom14: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    atom37_data = batched_gather(atom14, batch['residx_atom37_to_atom14'], dim=-2, no_batch_dims=len(atom14.shape[:-2]))
    atom37_data = atom37_data * batch['atom37_atom_exists'][..., None]
    return atom37_data