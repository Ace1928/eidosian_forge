import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def run_nodes(self, start: Optional[str]=None, end: Optional[str]=None):
    """
        Run part of the model from `start` node to `end` node. If `start` is None
        then we start from the beginning of the model. If `end` is None then we
        stop at the end of the model.

        Args:
            start: The name of the node which is the first node of the submodule
                we want to run. If set to None, then we'll start with the first
                node of the model.
            end: The name of the node which is the last node of the submodule we
                want to run. If set to None, we'll end with the last node of the
                model.
        """
    nodes = self._collect_nodes(start, end)
    cur_nodes = set(nodes)
    for node in nodes:
        if node in self.fusions:
            cur_nodes.update(self.fusions[node])
    output_names = []
    if self.settings.return_intermediate:
        output_names = [node.name for node in nodes]
    try:
        split_module, submod_name = self._build_submodule(cur_nodes)
        self._run_and_compare(split_module, submod_name, output_names)
    except (FxNetMinimizerRunFuncError, FxNetMinimizerResultMismatchError) as e:
        print(e)