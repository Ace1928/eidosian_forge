from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
def refresh_peers_(self, rotate: Optional[bool]=None) -> None:
    """Update in- and out-peers"""
    if rotate is None:
        rotate = self._graph_manager.is_dynamic_graph()
    assert not (rotate and (not self._graph_manager.is_dynamic_graph()))
    self.out_edges, self.in_edges = self._graph_manager.get_edges(rotate)