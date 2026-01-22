import inspect
import warnings
from typing import Any, List, Optional, Set
import torch
from torch.utils.data.datapipes.iter.sharding import (
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps
def get_all_graph_pipes(graph: DataPipeGraph) -> List[DataPipe]:
    return _get_all_graph_pipes_helper(graph, set())