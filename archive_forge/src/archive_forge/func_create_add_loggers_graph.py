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
def create_add_loggers_graph(model: GraphModule, subgraphs_dedup: Dict[str, List[Node]], qconfig_mapping: QConfigMapping, node_name_to_qconfig: Dict[str, QConfigAny]) -> None:
    """
    Given a model, a model graph partition (currently a set of matched
    subgraphs) and instructions how to transform each subgraph
    (currently quantizing it according to qconfig_mapping), modifies
    the model graph to create an alternate path through the original graph,
    with each of the subgraphs quantized.  This is useful to compare
    propagation error of a transformation such as quantization.

    For example, given layer op0 and op1, there are four cases when handling op1:
    1. op0 and op1 quantized
    2. op0 and op1 unquantized
    3. op0 quantized, op1 unquantized
    4. op0 unquantized, op1 quantized

    Example input, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \\                        \\          \\                 \\       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog    op1_1 -> x2_1 ----> clog

    Example output, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \\                        \\                           \\        # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_1 -> x2_1 ----> clog

    """
    from torch.ao.ns._numeric_suite_fx import OutputLogger, OutputComparisonLogger

    def _get_subgraph_containing_node(node, subgraphs_dedup):
        for subgraph in subgraphs_dedup.values():
            if node in subgraph:
                return subgraph
        return None
    nodes_to_skip = set()
    orig_first_node_to_shadow_in_node = {}
    orig_first_node_to_shadow_out_node = {}
    orig_nodes = list(model.graph.nodes)
    cur_subgraph_idx = 0
    for n in orig_nodes:
        if n.op in ('placeholder', 'get_attr', 'output') or n in nodes_to_skip:
            continue
        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        insert_submodule_copy = False
        if maybe_subgraph is not None:
            first_node, last_node = (maybe_subgraph[0], maybe_subgraph[-1])
            for node_to_skip in maybe_subgraph:
                nodes_to_skip.add(node_to_skip)
            qconfig = node_name_to_qconfig[first_node.name]
            if qconfig is not None:
                insert_submodule_copy = True
        else:
            first_node, last_node = (n, n)
        if insert_submodule_copy:
            match_name = first_node.name
            create_n_transformed_and_logged_copies_of_subgraph(model, cur_subgraph_idx, match_name, maybe_subgraph, [qconfig_mapping], [node_name_to_qconfig], None, None)
            expected_shadow_target = f'shadow_wrapper_{cur_subgraph_idx}_1'
            new_shadow_mod = None
            for maybe_shadow_mod in model.graph.nodes:
                if maybe_shadow_mod.op == 'call_module' and maybe_shadow_mod.target == expected_shadow_target:
                    new_shadow_mod = maybe_shadow_mod
                    break
            assert new_shadow_mod is not None
            orig_first_node_to_shadow_in_node[first_node] = new_shadow_mod
            orig_first_node_to_shadow_out_node[first_node] = new_shadow_mod
        else:
            subgraph_to_use = maybe_subgraph if maybe_subgraph is not None else [first_node]
            qconfig_str = ''
            subgraph_candidate_idx = 0
            fqn = _maybe_get_fqn(first_node, model)
            logger_mod_orig = _get_logger_for_subgraph(model, first_node, last_node, cur_subgraph_idx, subgraph_candidate_idx, qconfig_str, OutputLogger, fqn)
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            assert not hasattr(model, attr_name)
            setattr(model, attr_name, logger_mod_orig)
            insertion_point = last_node
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(attr_name, args=(last_node,), kwargs={})
                insertion_point = logger
            cur_node_orig = first_node
            cur_node_copy = None
            first_node_copy = None
            while cur_node_orig in subgraph_to_use:
                if cur_node_orig is first_node:
                    new_args = cur_node_orig.args
                    new_kwargs = cur_node_orig.kwargs
                else:
                    first_arg_for_copy = cur_node_copy
                    new_args = tuple([first_arg_for_copy, *cur_node_orig.args[1:]])
                    new_kwargs = cur_node_orig.kwargs
                with model.graph.inserting_after(insertion_point):
                    cur_node_copy = model.graph.create_node(cur_node_orig.op, cur_node_orig.target, new_args, new_kwargs)
                    if first_node_copy is None:
                        first_node_copy = cur_node_copy
                if cur_node_orig != last_node:
                    assert len(cur_node_orig.users.keys()) == 1
                cur_node_orig = next(iter(cur_node_orig.users.keys()))
                assert not cur_node_orig.name.startswith(SHADOW_NODE_NAME_PREFIX)
                insertion_point = cur_node_copy
            subgraph_candidate_idx = 1
            logger_mod_orig = _get_logger_for_subgraph(model, first_node, last_node, cur_subgraph_idx, subgraph_candidate_idx, qconfig_str, OutputComparisonLogger, fqn)
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            assert not hasattr(model, attr_name)
            setattr(model, attr_name, logger_mod_orig)
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(attr_name, args=(cur_node_copy, last_node), kwargs={})
            orig_first_node_to_shadow_in_node[first_node] = first_node_copy
            orig_first_node_to_shadow_out_node[first_node] = cur_node_copy
        cur_subgraph_idx += 1
    model.recompile()
    nodes_to_skip = set()
    for n in orig_nodes:
        if n.op in ('placeholder', 'get_attr', 'output') or n in nodes_to_skip:
            continue
        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        if maybe_subgraph is not None:
            first_node, last_node = (maybe_subgraph[0], maybe_subgraph[-1])
            for node_to_skip in maybe_subgraph:
                nodes_to_skip.add(node_to_skip)
        else:
            first_node, last_node = (n, n)

        def maybe_remap_node_to_shadow(node):
            """
            If unshadowed `node` has a shadow version, return that. If not,
            return `node`.
            """
            if not isinstance(node, Node):
                return node
            if node.op in ('placeholder', 'get_attr'):
                return node
            prev_subgraph = _get_subgraph_containing_node(node, subgraphs_dedup)
            if prev_subgraph is None:
                prev_subgraph = [node]
            prev_first_node = prev_subgraph[0]
            prev_shadow_output = orig_first_node_to_shadow_out_node[prev_first_node]
            return prev_shadow_output
        cur_shadow_input = orig_first_node_to_shadow_in_node[first_node]
        assert cur_shadow_input is not None
        cur_shadow_input.args = tree_map(maybe_remap_node_to_shadow, cur_shadow_input.args)
        cur_shadow_input.kwargs = tree_map(maybe_remap_node_to_shadow, cur_shadow_input.kwargs)
        model.recompile()