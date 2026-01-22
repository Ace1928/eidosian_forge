import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
def render_edge(self, edge: GraphEdge):
    attrs = {}
    if edge.headlabel is not None:
        attrs['headlabel'] = edge.headlabel
    if edge.taillabel is not None:
        attrs['taillabel'] = edge.taillabel
    if edge.kind is not None:
        if edge.kind == 'effect':
            attrs['style'] = 'dotted'
        elif edge.kind == 'meta':
            attrs['style'] = 'invis'
        elif edge.kind == 'cfg':
            attrs['style'] = 'solid'
            attrs['color'] = 'blue'
        else:
            raise ValueError(edge.kind)
    src = str(edge.src)
    dst = str(edge.dst)
    if edge.src_port:
        src += f':{edge.src_port}'
    if edge.dst_port:
        dst += f':{edge.dst_port}'
    self.digraph.edge(src, dst, **attrs)