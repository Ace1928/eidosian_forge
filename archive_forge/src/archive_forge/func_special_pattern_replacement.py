import torch
from torch.fx import map_arg, Node
from torch.fx.graph import Graph
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.fx import GraphModule
from .utils import (
from ..utils import _parent_name
from ..qconfig import QConfigAny
from ..quantization_mappings import get_quantized_operator
from .utils import create_node_from_old_node_preserve_meta
from typing import Dict, Tuple, Type, List, Callable, Any, Union, Set, Optional
import operator
def special_pattern_replacement(model: GraphModule):
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        q_node = n
        is_quantize = q_node.target == torch.quantize_per_tensor
        is_to_fp16 = q_node.op == 'call_method' and q_node.target == 'to' and (len(q_node.args) == 2) and (q_node.args[1] == torch.float16)
        if not (is_quantize or is_to_fp16):
            continue
        ref_node = q_node.args[0]
        is_call_function, is_call_method, is_call_module = is_fixed_qparams_node(ref_node, modules)
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            continue
        is_call_function, is_call_method, is_call_module = is_default_node(ref_node, modules)
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            continue
        is_call_function, is_call_method, is_call_module = is_special_pattern_node(ref_node, modules)
        if not (is_call_module or is_call_function or is_call_method):
            continue
        assert len(ref_node.args) > 0 or len(ref_node.kwargs) > 0
        dq_node_or_nodes = ref_node.args[0] if len(ref_node.args) > 0 else next(iter(ref_node.kwargs.values()))
        assert isinstance(dq_node_or_nodes, (Node, tuple, list))
        is_dequantize = False
        if isinstance(dq_node_or_nodes, Node):
            is_dequantize = dq_node_or_nodes.op == 'call_method' and dq_node_or_nodes.target == 'dequantize'
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            is_dequantize = all((x.op == 'call_method' and x.target == 'dequantize' for x in dq_node_or_nodes))
        if not is_dequantize:
            continue
        if is_call_module:
            ref_module = modules[ref_node.target]
            if type(ref_module) in SPECIAL_PATTERN_LOWER_MODULE_MAP and is_quantize:
                qmodule_cls = SPECIAL_PATTERN_LOWER_MODULE_MAP.get(type(ref_module))
                scale_node = q_node.args[1]
                zero_point_node = q_node.args[2]
                output_scale = getattr(model, scale_node.target)
                output_zero_point = getattr(model, zero_point_node.target)
                qmodule = qmodule_cls.from_reference(ref_module, output_scale, output_zero_point)
                parent_name, module_name = _parent_name(ref_node.target)
                setattr(modules[parent_name], module_name, qmodule)
        dq_nodes: List[Node] = []
        if isinstance(dq_node_or_nodes, Node):
            dq_nodes = [dq_node_or_nodes]
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            dq_nodes = list(dq_node_or_nodes)
        for dq_node in dq_nodes:
            dn_input = dq_node.args[0]
            ref_node.replace_input_with(dq_node, dn_input)
        qnode_qparams = list(q_node.args)[1:]
        q_node_input = q_node.args[0]
        q_node.replace_all_uses_with(q_node_input)
        model.graph.erase_node(q_node)
        is_call_function, is_call_method, is_call_module = is_default_node(ref_node, modules)
        if is_call_function:
            qop = get_quantized_operator(ref_node.target)
            args = list(ref_node.args)
            kwargs = dict(ref_node.kwargs)
            if qop in QOP_TO_ARG_NAMES_TO_SKIP:
                args_to_skip = QOP_TO_ARG_NAMES_TO_SKIP[qop]
                for arg in args_to_skip:
                    if arg in kwargs:
                        kwargs.pop(arg)
            kwargs['output_scale'] = qnode_qparams[0]
            kwargs['output_zero_point'] = qnode_qparams[1]
            with model.graph.inserting_after(qnode_qparams[1]):
                qop_node = create_node_from_old_node_preserve_meta(model.graph, ('call_function', qop, tuple(args), kwargs), ref_node)
                ref_node.replace_all_uses_with(qop_node)
                model.graph.erase_node(ref_node)
        else:
            for n in qnode_qparams:
                if isinstance(n, Node):
                    model.graph.erase_node(n)
    return model