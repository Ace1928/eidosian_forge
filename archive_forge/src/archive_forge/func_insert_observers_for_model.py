import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def insert_observers_for_model(model: GraphModule, node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig], node_name_to_qconfig: Dict[str, QConfigAny], prepare_custom_config: PrepareCustomConfig, equalization_config_map: Dict[str, Any], backend_config: BackendConfig, observed_node_names: Set[str], is_qat: bool) -> Optional[Node]:
    """
    Inserts observers, using the following high level algorithm:

    For each node in the graph:
      1. determine the target dtype of this node in the quantized graph, and save
           it for future steps
      2. determine the target dtype or all args and kwargs of this node
      3. if any arg or kwarg's target dtype does not match the current node's
           dtype, insert an observer
      4. if the current node needs an output observer, insert it

    For example:

    - starting graph:
        x0 -> linear -> x1

    - observed graph after processing x0:
        x0(fp32)

    - observed graph after processing linear:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8)

    - observed graph after processing x1:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8) -> x1

    After a node is processed, the naive observer placement is guaranteed to be
    complete for that node and all of its predecessors. There can be future
    passes which optimize the graph by deduplicating observers, etc.
    """
    cache_for_no_tensor_check: Dict[Node, bool] = {}
    named_modules = dict(model.named_modules(remove_duplicate=False))
    input_quantized_idxs: List[int] = prepare_custom_config.input_quantized_indexes
    output_quantized_idxs: List[int] = prepare_custom_config.output_quantized_indexes
    processed_nodes: Set[Node] = set()
    for node in model.graph.nodes:
        node.meta['target_dtype_info'] = copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)
    inputs_seen_counter = 0
    outputs_seen_counter = 0
    placeholder_node_to_input_index: Dict[Node, int] = {}
    output_node_to_output_index: Dict[Node, int] = {}
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            placeholder_node_to_input_index[node] = inputs_seen_counter
            inputs_seen_counter += 1
        if node.op == 'output':
            output_node_to_output_index[node] = outputs_seen_counter
            outputs_seen_counter += 1
    for match_res_with_qconfig in node_name_to_match_result_with_qconfig.values():
        last_node, matched_node_pattern, pattern, qhandler, qconfig = match_res_with_qconfig
        assert qhandler is not None
        _set_target_dtype_info_for_matched_node_pattern(matched_node_pattern, last_node, qconfig, qhandler, backend_config, named_modules, cache_for_no_tensor_check, processed_nodes)
    for node in model.graph.nodes:
        if node.op == 'placeholder' and placeholder_node_to_input_index[node] in input_quantized_idxs:
            node.meta['target_dtype_info'] = copy.copy(_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO)
        elif node.op in ('call_module', 'call_method', 'call_function'):
            args_have_no_tensors = all_node_args_have_no_tensors(node, named_modules, cache_for_no_tensor_check)
            if args_have_no_tensors:
                node.meta['target_dtype_info'] = {'input_act_obs_or_fq_ctr': None, 'output_act_obs_or_fq_ctr': None}
        elif node.op == 'output' and output_node_to_output_index[node] in output_quantized_idxs:
            node.meta['target_dtype_info'] = copy.copy(_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO)
    propagate_dtypes_for_known_nodes(model.graph, node_name_to_match_result_with_qconfig)
    processed_nodes: Set[Node] = set()
    for match_res_with_qconfig in node_name_to_match_result_with_qconfig.values():
        last_node, matched_node_pattern, pattern, qhandler, qconfig = match_res_with_qconfig
        is_supported_by_backend = _is_pattern_dtype_config_and_qconfig_supported_by_backend(pattern, matched_node_pattern, qconfig, backend_config)
        assert qhandler is not None
        output_act_or_fq_ctr = node.meta['target_dtype_info']['output_act_obs_or_fq_ctr']
        output_act_or_fq = output_act_or_fq_ctr() if output_act_or_fq_ctr else None
        output_act_dtype, _ = _get_dtype_and_is_dynamic(output_act_or_fq)
        if not is_supported_by_backend and output_act_dtype not in [None, int, float, torch.bool]:
            _set_target_dtype_info_for_matched_node_pattern(matched_node_pattern, last_node, torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig, None, backend_config, named_modules, cache_for_no_tensor_check, processed_nodes)
    nodes_before_observation = list(model.graph.nodes)
    custom_module_names_already_swapped: Set[str] = set()
    inputs_seen_counter = 0
    outputs_seen_counter = 0
    results_node = None
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    for node in nodes_before_observation:
        if node.op == 'placeholder':
            pass
        elif node.op in ('call_module', 'call_method', 'call_function', 'output'):
            last_node, matched_node_pattern, pattern, qhandler, qconfig = node_name_to_match_result_with_qconfig.get(node.name, (None, None, None, None, None))
            equalization_qconfig = equalization_config_map.get(node.name, None)
            this_node_dtype_info = node.meta['target_dtype_info']
            if 'val' in node.meta:
                output_is_a_tensor = this_node_dtype_info is not None and isinstance(node.meta['val'], FakeTensor)
            else:
                output_is_a_tensor = this_node_dtype_info is not None
            skip_inserting_observers = (qconfig is None or not output_is_a_tensor) and (not node.op == 'output')
            is_supported_by_backend = _is_pattern_dtype_config_and_qconfig_supported_by_backend(pattern, matched_node_pattern, qconfig, backend_config)
            if not skip_inserting_observers and is_supported_by_backend:
                named_modules = dict(model.named_modules(remove_duplicate=False))
                if node.op != 'output':
                    assert matched_node_pattern is not None
                    _add_matched_node_name_to_set(matched_node_pattern, observed_node_names)
                    is_quantized_branch = False
                    if len(node.args) > 0 and isinstance(node.args[0], Node) and (len(node.args[0].users) > 1):
                        for user in node.args[0].users:
                            is_user_quantized = node_name_to_qconfig.get(user.name, None) is not None or (user.op == 'call_module' and isinstance(named_modules[str(user.target)], ObserverBase))
                            if user != node and is_user_quantized:
                                is_quantized_branch = True
                    pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)
                    root_node_getter = pattern_to_root_node_getter.get(pattern, _default_root_node_getter)
                    root_node = root_node_getter(matched_node_pattern)
                    is_input_node_of_the_pattern = node is root_node
                    if is_input_node_of_the_pattern:
                        _maybe_insert_input_observers_for_node(node, qconfig, model, named_modules, model.graph, qhandler, prepare_custom_config, obs_or_fq_map, is_qat, backend_config)
                        _maybe_insert_input_equalization_observers_for_node(node, equalization_qconfig, model, named_modules, model.graph, is_quantized_branch)
                    is_last_node_of_pattern = node is last_node
                    input_output_share_observers = node.meta['target_dtype_info'].get('input_output_share_observers', False)
                    reuse_input_obs_or_fq = node.meta['target_dtype_info'].get('reuse_input_obs_or_fq', False)
                    if is_last_node_of_pattern:
                        if _is_custom_module_lstm(node, named_modules, qconfig, qhandler):
                            _insert_dequant_stubs_for_custom_module_lstm_output(node, model, named_modules, model.graph)
                            if node.target not in custom_module_names_already_swapped:
                                custom_module_names_already_swapped.add(node.target)
                                _swap_custom_module_to_observed(node, qconfig, named_modules, prepare_custom_config)
                        else:
                            maybe_output_obs_node = _maybe_insert_output_observer_for_node(node, model, named_modules, model.graph, obs_or_fq_map, is_qat)
                            if maybe_output_obs_node is not None:
                                orig_users = list(node.users.keys())
                                for user_node in orig_users:
                                    if user_node is maybe_output_obs_node:
                                        continue
                                    user_node.replace_input_with(node, maybe_output_obs_node)
                                _is_observer_in_same_graph_ = _is_observer_in_same_graph(node, named_modules, obs_or_fq_map, is_qat)
                                if input_output_share_observers and _is_observer_in_same_graph_ or reuse_input_obs_or_fq:
                                    if not _maybe_make_input_output_share_observers(node, model, named_modules):
                                        _remove_output_observer(node, model, named_modules)
                                if qhandler is not None and qhandler.is_custom_module():
                                    if node.target not in custom_module_names_already_swapped:
                                        custom_module_names_already_swapped.add(node.target)
                                        _swap_custom_module_to_observed(node, qconfig, named_modules, prepare_custom_config)
                else:
                    _maybe_insert_observers_before_graph_output(node, model, named_modules, model.graph, obs_or_fq_map, is_qat)
        if node.op == 'placeholder':
            inputs_seen_counter += 1
        elif node.op == 'output':
            outputs_seen_counter += 1
            results_node = node
    return results_node