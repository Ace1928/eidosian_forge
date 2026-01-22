import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def extract_weight_comparison(m: GraphModule) -> NSResultsType:
    weighted_ops = {torch.nn.functional.linear}
    results: NSResultsType = {'model': {NSSingleResultValuesType.WEIGHT.value: {}}}
    for n in m.graph.nodes:
        if not (n.op == 'call_function' and n.target in weighted_ops):
            continue
        first_arg = n.args[0]
        shadow_wrapper_node = None
        for user in first_arg.users:
            if user.op == 'call_module' and user.target.startswith('shadow_wrapper'):
                shadow_wrapper_node = user
                break
        if shadow_wrapper_node is None:
            continue
        shadow_wrapper = getattr_from_fqn(m, shadow_wrapper_node.target)
        weight_info = _get_weight_info_from_shadow_wrapper(shadow_wrapper)
        if weight_info is None:
            continue
        w_node = n.args[1]
        w_obj = getattr_from_fqn(m, w_node.target).detach()
        quant_fn, quant_fn_args_except_first = weight_info
        new_args = (w_obj, *quant_fn_args_except_first)
        w_obj_q = quant_fn(*new_args)
        ref_node_name = n.name
        prev_node_name = n.name
        ref_node_type = get_target_type_str(n, m)
        prev_node_type = ref_node_type
        fqn = None
        if hasattr(m, '_node_name_to_scope'):
            fqn = m._node_name_to_scope[n.name][0]
        comparison = torch.ao.ns.fx.utils.compute_sqnr(w_obj, w_obj_q)
        result_fp32 = {'res_type': NSSingleResultValuesType.WEIGHT.value, 'values': [w_obj], 'prev_node_name': prev_node_name, 'prev_node_target_type': prev_node_type, 'ref_node_name': ref_node_name, 'ref_node_target_type': ref_node_type, 'index_within_arg': 0, 'index_of_arg': 0, 'fqn': fqn, 'qconfig_str': '', 'comparisons': [comparison], 'comparison_fn_name': 'sqnr'}
        result_q = {'res_type': NSSingleResultValuesType.WEIGHT.value, 'values': [w_obj_q], 'prev_node_name': prev_node_name, 'prev_node_target_type': prev_node_type, 'ref_node_name': ref_node_name, 'ref_node_target_type': ref_node_type, 'index_within_arg': 0, 'index_of_arg': 0, 'fqn': fqn, 'qconfig_str': '', 'comparisons': [comparison], 'comparison_fn_name': 'sqnr'}
        _1, _2, node_idx, _3 = shadow_wrapper_node.target.split('_')
        name_fp32 = f'subgraph_{node_idx}_0'
        name_q = f'subgraph_{node_idx}_1'
        results['model'][NSSingleResultValuesType.WEIGHT.value][name_fp32] = [result_fp32]
        results['model'][NSSingleResultValuesType.WEIGHT.value][name_q] = [result_q]
    return results