import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
def render_graph(self, builder: 'GraphBuilder'):
    reached_vs: set[ValueState] = set()
    for vs in [*self.out_vars.values(), _just(self.out_effect)]:
        self._gather_reachable(vs, reached_vs)
    reached_vs.add(_just(self.in_effect))
    reached_vs.update(self.in_vars.values())
    reached_op = {vs.parent for vs in reached_vs if vs.parent is not None}
    for vs in reached_vs:
        self._render_vs(builder, vs)
    for op in reached_op:
        self._render_op(builder, op)
    ports = []
    outgoing_nodename = f'outgoing_{self.name}'
    for k, vs in self.out_vars.items():
        ports.append(k)
        builder.graph.add_edge(vs.short_identity(), outgoing_nodename, dst_port=k)
    outgoing_node = builder.node_maker.make_node(kind='ports', ports=ports, data=dict(body='outgoing'))
    builder.graph.add_node(outgoing_nodename, outgoing_node)
    ports = []
    incoming_nodename = f'incoming_{self.name}'
    for k, vs in self.in_vars.items():
        ports.append(k)
        builder.graph.add_edge(incoming_nodename, _just(vs.parent).short_identity(), src_port=k)
    incoming_node = builder.node_maker.make_node(kind='ports', ports=ports, data=dict(body='incoming'))
    builder.graph.add_node(incoming_nodename, incoming_node)
    jt_node = builder.node_maker.make_node(kind='meta', data=dict(body=f'jump-targets: {self._jump_targets}'))
    builder.graph.add_node(f'jt_{self.name}', jt_node)