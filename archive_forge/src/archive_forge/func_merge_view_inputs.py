import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
from .subclass_utils import (
from .utils import (
def merge_view_inputs(fwd_inputs: List[Any], mutated_input_info: List[InputAliasInfo], *, is_inference: bool) -> Tuple[List[Any], Optional[List[Union[int, Tuple[int, torch.Tensor]]]]]:

    def _are_differentiable_views(view1, view2):
        if view1 is view2:
            return True
        if view1._base is None and view2._base is None:
            return False
        if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
            return True
        return False

    def _same_dtype_views(view1, view2):
        if view1.dtype != view2.dtype:
            return False
        if view1._base is not None and view1.dtype != view1._base.dtype:
            return False
        if view2._base is not None and view2.dtype != view2._base.dtype:
            return False
        return True
    assert len(fwd_inputs) == len(mutated_input_info)
    storage_ref_to_idx: Dict[StorageWeakRef, List[int]] = collections.defaultdict(list)
    base_args = []
    other_args = []
    for i, inpt in enumerate(fwd_inputs):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt.untyped_storage())
            storage_ref_to_idx[storage_ref].append(i)
        else:
            other_args.append(inpt)
    inner_calling_convention_meta: Dict[int, Union[int, Tuple[int, torch.Tensor]]] = {}
    for aliased_input_indices in storage_ref_to_idx.values():
        if len(aliased_input_indices) <= 1 or not any((mutated_input_info[inpt_idx].mutates_data for inpt_idx in aliased_input_indices)):
            for curr_idx in aliased_input_indices:
                other_args.append(fwd_inputs[curr_idx])
            continue
        aliased_input_indices_no_false_sharing = compute_overlapping_inputs(fwd_inputs, aliased_input_indices)
        if len(aliased_input_indices_no_false_sharing) <= 1:
            for curr_idx in aliased_input_indices:
                other_args.append(fwd_inputs[curr_idx])
            continue
        for idx1, idx2 in zip(aliased_input_indices, aliased_input_indices[1:], strict=False):
            view1 = fwd_inputs[idx1]
            view2 = fwd_inputs[idx2]
            if not is_inference:
                assert _are_differentiable_views(view1, view2), 'aot_autograd() does not yet handle non-differentiable view input mutations.'
            assert _same_dtype_views(view1, view2), 'aot_autograd() does not yet handle input mutations on views with different dtypes.'
        non_none_bases = [fwd_inputs[i]._base for i in aliased_input_indices if fwd_inputs[i]._base is not None]
        aliases_with_none_bases = [fwd_inputs[i] for i in aliased_input_indices if fwd_inputs[i]._base is None]
        if len(non_none_bases) == 0:
            example_idx = aliased_input_indices[0]
            example_alias = fwd_inputs[example_idx]
            synthetic_base = torch.empty((0,), dtype=example_alias.dtype, device=example_alias.device)
            synthetic_base.set_(example_alias.untyped_storage())
        else:
            synthetic_base = non_none_bases[0]
            for other_base in non_none_bases[1:]:
                assert other_base is synthetic_base, 'aot_autograd() does not yet handle non-differentiable view input mutations.'
            for alias in aliases_with_none_bases:
                assert alias is synthetic_base, 'aot_autograd() does not yet handle non-differentiable view input mutations.'
        base_args.append(synthetic_base)
        for curr_view_idx in aliased_input_indices:
            curr_view = fwd_inputs[curr_view_idx]
            base_idx = len(base_args) - 1
            inner_calling_convention_meta[curr_view_idx] = (base_idx, curr_view)
    if len(base_args) == 0:
        assert len(other_args) == len(fwd_inputs)
        return (fwd_inputs, None)
    else:
        args_to_functionalization = base_args + other_args
        arg_to_old_idx_map = {arg: i for i, arg in enumerate(fwd_inputs)}
        for i, other_arg in enumerate(other_args):
            new_idx = len(base_args) + i
            old_idx = arg_to_old_idx_map[other_arg]
            inner_calling_convention_meta[old_idx] = new_idx
        post_processed_calling_convention_meta: List[Union[int, Tuple[int, torch.Tensor]]] = [-1 for _ in range(len(inner_calling_convention_meta))]
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        for x in post_processed_calling_convention_meta:
            assert x != -1
        return (args_to_functionalization, post_processed_calling_convention_meta)