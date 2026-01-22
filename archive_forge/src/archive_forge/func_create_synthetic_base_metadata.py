import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from .functional_utils import _get_mutation_type
from .schemas import (
from .utils import strict_zip
def create_synthetic_base_metadata(m: ViewAndMutationMeta, synthetic_base_info: List[Union[int, Tuple[int, torch.Tensor]]], outer_args: List[Any], inner_args: List[Any]) -> Tuple[ViewAndMutationMeta, List[int]]:
    synthetic_base_to_indices: Dict[int, List[int]] = {}
    for inner_idx in range(len(inner_args)):
        outer_aliased_indices_of_current_base_arg = [outer_idx for outer_idx, inner_idx_or_tuple in enumerate(synthetic_base_info) if isinstance(inner_idx_or_tuple, int) and inner_idx_or_tuple == inner_idx or (isinstance(inner_idx_or_tuple, tuple) and inner_idx_or_tuple[0] == inner_idx)]
        synthetic_base_to_indices[inner_idx] = outer_aliased_indices_of_current_base_arg
    input_infos = []
    for outer_indices in synthetic_base_to_indices.values():
        any_leaf = any((m.input_info[x].is_leaf for x in outer_indices))
        all_leaf = all((m.input_info[x].is_leaf for x in outer_indices))
        assert any_leaf == all_leaf
        mutates_data = True if len(outer_indices) > 1 else m.input_info[outer_indices[0]].mutates_data
        mutates_metadata = False if len(outer_indices) > 1 else m.input_info[outer_indices[0]].mutates_metadata
        requires_grad = any((m.input_info[x].requires_grad for x in outer_indices))
        mutations_hidden_from_autograd = all((m.input_info[x].mutations_hidden_from_autograd for x in outer_indices))
        mutations_under_no_grad_or_inference_mode = all((m.input_info[x].mutations_under_no_grad_or_inference_mode for x in outer_indices))
        mutation_type = _get_mutation_type(m.keep_input_mutations, mutates_data, mutates_metadata, mutations_hidden_from_autograd, mutations_under_no_grad_or_inference_mode, requires_grad)
        inpt_info = InputAliasInfo(mutates_data=mutates_data, mutates_metadata=mutates_metadata, mutations_hidden_from_autograd=all((m.input_info[x].mutations_hidden_from_autograd for x in outer_indices)), mutates_storage_metadata=False if len(outer_indices) > 1 else m.input_info[outer_indices[0]].mutates_storage_metadata, mutations_under_no_grad_or_inference_mode=mutations_under_no_grad_or_inference_mode, is_leaf=any_leaf, requires_grad=requires_grad, mutation_type=mutation_type)
        input_infos.append(inpt_info)
    outer_aliased_arg_idx_with_metadata_mutations = [outer_idx for outer_idx, inpt_info in enumerate(m.input_info) if inpt_info.mutates_metadata and (not isinstance(synthetic_base_info[outer_idx], int))]
    input_metadata_output_info = [OutputAliasInfo(output_type=OutputType.alias_of_input, raw_type=FunctionalTensor, dynamic_dims={i for i, s in enumerate(outer_args[outer_idx].shape) if not is_concrete_int(s)}, base_idx=synthetic_base_info[outer_idx][0], requires_grad=outer_args[outer_idx].requires_grad) for outer_idx in outer_aliased_arg_idx_with_metadata_mutations]
    existing_output_infos = [OutputAliasInfo(output_type=o.output_type, raw_type=o.raw_type, dynamic_dims=o.dynamic_dims, base_idx=None if o.base_idx is None else synthetic_base_info[o.base_idx] if isinstance(synthetic_base_info[o.base_idx], int) else synthetic_base_info[o.base_idx][0], requires_grad=o.requires_grad) for o in m.output_info]
    inner_mutated_tangents = [x for inner_idx, x in enumerate(inner_args) if input_infos[inner_idx].mutates_data and input_infos[inner_idx].requires_grad]
    output_info = existing_output_infos + input_metadata_output_info
    traced_tangents = inner_mutated_tangents + m.traced_tangents[len(inner_mutated_tangents):]
    return (ViewAndMutationMeta(input_info=input_infos, output_info=output_info, num_intermediate_bases=m.num_intermediate_bases, keep_input_mutations=m.keep_input_mutations, traced_tangents=traced_tangents, subclass_inp_meta=[], subclass_fw_graph_out_meta=[], subclass_tangent_meta=[], is_train=m.is_train), outer_aliased_arg_idx_with_metadata_mutations)