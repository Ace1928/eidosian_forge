import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
def to_graphviz(graph: GraphBacking):
    """Render a GraphBacking using graphviz
    """
    rgr = GraphvizRendererBackend()
    graph.render(rgr)
    return rgr.digraph