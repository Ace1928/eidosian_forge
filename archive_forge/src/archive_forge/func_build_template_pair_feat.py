from typing import Dict, Tuple, overload
import torch
import torch.types
from torch import nn
from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather
def build_template_pair_feat(batch: Dict[str, torch.Tensor], min_bin: torch.types.Number, max_bin: torch.types.Number, no_bins: int, use_unit_vector: bool=False, eps: float=1e-20, inf: float=100000000.0) -> torch.Tensor:
    template_mask = batch['template_pseudo_beta_mask']
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    tpb = batch['template_pseudo_beta']
    dgram = torch.sum((tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, no_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
    to_concat = [dgram, template_mask_2d[..., None]]
    aatype_one_hot: torch.LongTensor = nn.functional.one_hot(batch['template_aatype'], rc.restype_num + 2)
    n_res = batch['template_aatype'].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(*aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[..., None, :].expand(*aatype_one_hot.shape[:-2], -1, n_res, -1))
    n, ca, c = [rc.atom_order[a] for a in ['N', 'CA', 'C']]
    rigids = Rigid.make_transform_from_reference(n_xyz=batch['template_all_atom_positions'][..., n, :], ca_xyz=batch['template_all_atom_positions'][..., ca, :], c_xyz=batch['template_all_atom_positions'][..., c, :], eps=eps)
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)
    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec ** 2, dim=-1))
    t_aa_masks = batch['template_all_atom_mask']
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    if not use_unit_vector:
        unit_vector = unit_vector * 0.0
    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])
    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]
    return act