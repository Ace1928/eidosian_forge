from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
def reordering_to_mimic_autograd_engine(gm):
    """
    This pass finds the first bwd node in the graph (by looking at users of
    tangents) and then reorders the graph by walking from this node to all the
    way to the end of the graph. At each op in this traveral, we insert this op
    in a new graph and try to bring only the relevant subgraph from the other
    non-bwd edges relevant for this op. This closely mimics the behavior of
    autograd engine.

    Why is this pass required in the first place?

    This is an artifact of how partitioners work today. The starting point of
    partitioner is a joint graph, which is fwd and then bwd graph. In the case
    of checkpointing, we keep portions of fwd graph in their original place in
    the joint graph, while obtaining a bwd graph. As a result, the resulting bwd
    graph has copies of recomputed fwd subgraphs followed by the original bwd
    graph. If we run this naively, this leads to bad memory footprint, because
    the fwd subgraphs are live for way longer duration than necessary. This pass
    reorders the operations such that we prioritize the ops for the original bwd
    graph while only realizing those ops from the fwd graph that are necessary
    at any given point in the graph.
    """
    new_graph = fx.Graph()
    env = {}
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            new_node = new_graph.placeholder(node.name)
            new_node.meta = node.meta
            env[node] = new_node
    order = {}
    for idx, node in enumerate(gm.graph.nodes):
        order[node] = idx
    depths = {}
    output_node = next((node for node in gm.graph.nodes if node.op == 'output'))
    get_depth(output_node, depths)

    def insert_node_in_graph(node):
        if node in env:
            return env[node]
        for arg, _ in sort_depths(node.all_input_nodes, depths):
            env[arg] = insert_node_in_graph(arg)
        env[node] = new_graph.node_copy(node, lambda x: env[x])
        return env[node]
    tangent_inputs = list(filter(_is_tangent, gm.graph.nodes))
    first_node_in_bwd = None
    minimum_order = math.inf
    for tangent in tangent_inputs:
        for user in tangent.users:
            if order[user] < minimum_order:
                minimum_order = order[user]
                first_node_in_bwd = user
    assert first_node_in_bwd is not None
    for node in list(gm.graph.nodes)[order[first_node_in_bwd]:]:
        insert_node_in_graph(node)
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm