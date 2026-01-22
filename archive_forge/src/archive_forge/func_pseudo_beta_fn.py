from typing import Dict, Tuple, overload
import torch
import torch.types
from torch import nn
from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather
def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_gly = aatype == rc.restype_order['G']
    ca_idx = rc.atom_order['CA']
    cb_idx = rc.atom_order['CB']
    pseudo_beta = torch.where(is_gly[..., None].expand(*(-1,) * len(is_gly.shape), 3), all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        return (pseudo_beta, pseudo_beta_mask)
    else:
        return pseudo_beta