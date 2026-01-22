import inspect
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from collections import OrderedDict
import logging
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
def construct_graph(node: Node, base_mod_env: Dict[str, Node], base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule]):
    if node.op == 'placeholder':
        default_value = node.args[0] if len(node.args) > 0 else inspect.Signature.empty
        base_mod_env[node.name] = base_mod_graph.placeholder(node.target, type_expr=node.type, default_value=default_value)
        base_mod_env[node.name].meta = node.meta.copy()
    elif node.op == 'get_attr':
        base_mod_env[node.name] = base_mod_graph.get_attr(node.target)
        base_mod_env[node.name].meta = node.meta.copy()
        attr_val = m
        for atom in node.target.split('.'):
            if not hasattr(attr_val, atom):
                raise AttributeError(f'Node target {node.target} not found!')
            attr_val = getattr(attr_val, atom)
        base_mod_attrs[node.target] = attr_val
    return (base_mod_env, base_mod_attrs)