import collections
import contextlib
import cProfile
import dataclasses
import functools
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
import torch
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._pytree import tree_map
from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
from .virtualized import V
from torch._inductor.debug import load_args_and_run_compile_fx_inner
def create_fx_from_snodes(snodes: List[BaseSchedulerNode]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """

    def get_fake_func(name):

        def func1(*args):
            return 0
        func1.__name__ = name
        return func1
    FusionMeta = collections.namedtuple('FusionMeta', ['group', 'snode', 'type'])
    buf_to_fx_node = {}
    graph = torch.fx.Graph()
    first_node = None
    outputs = []
    group: Any = None
    for snode in snodes:
        if snode.is_extern():
            node_type = 'extern'
            group = node_type
        elif snode.is_template():
            node_type = 'template'
            group = node_type
        elif isinstance(snode, NopKernelSchedulerNode):
            node_type = 'nop'
            group = node_type
        elif isinstance(snode, SchedulerNode):
            node_type = 'compute'
            group = snode.group
        elif isinstance(snode, FusedSchedulerNode):
            node_type = 'fused'
            group = snode.group
        else:
            raise RuntimeError('Unknown node type')
        fused_name = torch._inductor.utils.get_fused_kernel_name(snode.get_nodes(), 'original_aten')
        func_name = f'{node_type}: {fused_name}'
        node_func = get_fake_func(func_name)
        kwargs = {}
        if hasattr(snode, 'get_device'):
            kwargs = {'device': snode.get_device()}
        fx_node = graph.call_function(node_func, args=(), kwargs=kwargs)

        def in_output(snode):
            if isinstance(snode, FusedSchedulerNode):
                return any((in_output(x) for x in snode.snodes))
            return any((isinstance(user.node, OutputNode) for user in snode.users))
        if in_output(snode):
            outputs.append(fx_node)
        name = snode.get_name()
        fx_node.name = name
        fx_node.meta['fusion_meta'] = FusionMeta(group, snode, node_type)
        if isinstance(snode, FusedSchedulerNode):
            for x in snode.snodes:
                buf_to_fx_node[x.get_name()] = fx_node
        buf_to_fx_node[name] = fx_node
        if first_node is None:
            first_node = fx_node
    for snode in snodes:
        name = snode.get_name()
        deps = snode.read_writes.reads
        fx_node = buf_to_fx_node[name]
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            new_args.append(dep_node)
        fx_node.args = tuple(new_args)
    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    return graph