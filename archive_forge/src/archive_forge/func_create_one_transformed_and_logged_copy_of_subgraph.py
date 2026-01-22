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
def create_one_transformed_and_logged_copy_of_subgraph(mt: GraphModule, subgraph_idx: int, subgraph_candidate_idx: int, first_node: Node, last_node: Node, fqn: Optional[str], list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]], example_inputs: Any, last_added_shadow_node_list: List[Optional[Node]], custom_prepare_fn: Optional[Callable]=None, custom_prepare_kwargs: Optional[Dict[str, Any]]=None) -> None:
    """
    Given a subgraph in `mt` and a subgraph candidate idx, inserts the
    subgraph candidate copy and instruments it with loggers.

    If subgraph_candidate_idx is 0, this is the baseline fp32 subgraph and we just
    add a logger to the end.

    If subgraph_candidate_idx is not 0, we create a copy of the subgraph and
    prepare it with `prepare_fx`.
    """
    from torch.ao.ns._numeric_suite_fx import OutputLogger, OutputComparisonLogger
    if subgraph_candidate_idx == 0:
        qconfig_str = ''
        logger_mod_orig = _get_logger_for_subgraph(mt, first_node, last_node, subgraph_idx, subgraph_candidate_idx, qconfig_str, OutputLogger, fqn)
        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        assert not hasattr(mt, attr_name)
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(last_node):
            new_node = mt.graph.call_module(attr_name, args=(last_node,), kwargs={})
            last_added_shadow_node_list[0] = new_node
    else:
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]
        if qconfig is None:
            return
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        orig_mod_copy_wrapped = create_submodule_from_subgraph(mt, first_node, last_node)
        if custom_prepare_fn is None:
            orig_mod_copy_wrapped = torch.ao.quantization.quantize_fx.prepare_fx(orig_mod_copy_wrapped, qconfig_mapping, example_inputs=example_inputs)
        else:
            if custom_prepare_kwargs is None:
                custom_prepare_kwargs = {}
            for kwarg_name in ['example_inputs', 'prepare_custom_config', 'qconfig_mapping']:
                assert kwarg_name not in custom_prepare_kwargs, f'cannot specify {kwarg_name} in custom_prepare_kwargs'
            prepare_kwargs: Dict[str, Any] = {'example_inputs': example_inputs, 'qconfig_mapping': qconfig_mapping}
            prepare_kwargs.update(custom_prepare_kwargs)
            orig_mod_copy_wrapped = custom_prepare_fn(orig_mod_copy_wrapped, **prepare_kwargs)
        attr_name = _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx)
        assert not hasattr(mt, attr_name)
        setattr(mt, attr_name, orig_mod_copy_wrapped)
        insert_after_node = last_added_shadow_node_list[0]
        with mt.graph.inserting_after(insert_after_node):
            new_args = []
            for arg in first_node.args:
                if isinstance(arg, Node):
                    new_args.append(arg)
                elif isinstance(arg, (list, tuple)) and len(arg) and isinstance(arg[0], Node):
                    for inner_arg in arg:
                        if isinstance(inner_arg, Node):
                            new_args.append(inner_arg)
            new_kwargs = {}
            for name, old_kwarg in first_node.kwargs.items():
                if isinstance(old_kwarg, Node):
                    new_kwargs[name] = old_kwarg
                elif isinstance(old_kwarg, (list, tuple)) and len(old_kwarg):
                    for inner_old_kwarg in old_kwarg:
                        new_args.append(inner_old_kwarg)
            new_args = tuple(new_args)
            new_node = mt.graph.call_module(attr_name, args=new_args, kwargs=new_kwargs)
        logger_mod_orig = _get_logger_for_subgraph(mt, first_node, last_node, subgraph_idx, subgraph_candidate_idx, str(qconfig), OutputComparisonLogger, fqn)
        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        assert not hasattr(mt, attr_name)
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(new_node):
            logger = mt.graph.call_module(attr_name, args=(new_node, last_node), kwargs={})
            last_added_shadow_node_list[0] = logger
    mt.recompile()