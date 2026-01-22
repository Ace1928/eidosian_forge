import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
def compute_dependencies(self):
    """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """
    name_to_users: DefaultDict[str, List[NodeUser]] = collections.defaultdict(list)
    for node1 in self.nodes:
        node1_name = node1.get_name()
        for node2_name in node1.get_aliases():
            if node1_name in name_to_users and node2_name in name_to_users:
                list1 = name_to_users[node1_name]
                list2 = name_to_users[node2_name]
                combined = list1 + list2
                for key in name_to_users.keys():
                    if name_to_users[key] is list1 or name_to_users[key] is list2:
                        name_to_users[key] = combined
            elif node1_name in name_to_users:
                name_to_users[node2_name] = name_to_users[node1_name]
            else:
                name_to_users[node1_name] = name_to_users[node2_name]

    def rename(n):
        if n in self.mutation_renames:
            return rename(self.mutation_renames[n])
        return n

    def dep_closure(node_name):
        reachable_names = {node_name}
        node = self.name_to_node[node_name]
        write_dep = next(iter(node.read_writes.writes))
        for read_dep in node.read_writes.reads:
            if read_dep.name in self.name_to_node and isinstance(read_dep, dependencies.MemoryDep) and isinstance(write_dep, dependencies.MemoryDep) and (read_dep.index == write_dep.index) and (read_dep.size == write_dep.size):
                reachable_names.update(dep_closure(read_dep.name))
        return reachable_names

    def add_user(used_by_name, user_node, can_inplace=False, is_weak=False):
        name_to_users[rename(used_by_name)].append(NodeUser(user_node, can_inplace, is_weak))
    unbacked_symbol_to_origin_node = {}
    for node in self.nodes:
        log.debug('scheduling %s', node.node)
        for s in node.node.get_unbacked_symbol_defs():
            assert isinstance(s, sympy.Symbol)
            if s not in unbacked_symbol_to_origin_node:
                unbacked_symbol_to_origin_node[s] = node
        for s in node.node.get_unbacked_symbol_uses():
            assert s in unbacked_symbol_to_origin_node, f'{s} not in {unbacked_symbol_to_origin_node}'
            node.add_fake_dep(StarDep(unbacked_symbol_to_origin_node[s].get_name()))
        assert len(node.get_mutations()) <= 1
        for alt_name in node.get_mutations():
            alt_name = rename(alt_name)
            add_user(alt_name, node)
            node.add_mutation_dep(StarDep(alt_name))
            for other_node in name_to_users[alt_name]:
                other_name = rename(other_node.get_name())
                known_dep_node_names = dep_closure(node.get_name())
                if other_name not in known_dep_node_names:
                    node.add_mutation_dep(WeakDep(other_name))
                    add_user(other_name, node, is_weak=True)
        for read in node.read_writes.reads:
            is_weak = isinstance(read, WeakDep)
            add_user(read.name, node, node.can_inplace(read), is_weak)
        node.update_mutated_names(self.mutation_renames)
        for alt_name in node.get_mutations():
            self.mutation_renames[rename(alt_name)] = node.get_name()
            self.mutation_renames[alt_name] = node.get_name()
            self.mutation_real_name[node.get_name()] = self.mutation_real_name.get(alt_name, alt_name)
    for node_name in V.graph.get_output_names():
        log.debug('scheduling output %s', node_name)
        add_user(node_name, OutputNode(StarDep(node_name)))
    for node in V.graph.graph_outputs:
        if isinstance(node, ir.ShapeAsConstantBuffer):
            for s in free_unbacked_symbols(node.shape):
                assert s in unbacked_symbol_to_origin_node, f'{s} not in {unbacked_symbol_to_origin_node.keys()}'
                node_name = unbacked_symbol_to_origin_node[s].node.name
                log.debug('scheduling output %s for unbacked symint %s', node_name, s)
                add_user(node_name, OutputNode(StarDep(node_name)))
    for name in self.mutation_renames:
        if name in V.graph.graph_inputs:
            add_user(name, OutputNode(StarDep(name)))
            V.graph.mutated_inputs.add(name)
    inp_names = {name: index for index, name in enumerate(V.graph.graph_inputs.keys())}
    V.graph.mutated_input_idxs = [inp_names[name] for name in V.graph.mutated_inputs]
    for node in self.nodes:
        node.set_users(name_to_users[node.get_name()])
    for node in self.nodes:
        for user in node.users:
            user.node.inverse_users.append(node)