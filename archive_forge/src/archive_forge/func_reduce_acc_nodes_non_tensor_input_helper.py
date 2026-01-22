import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList):
    """
        Transitively excludes nodes from ACC supported set.
        For every node in the worklist:
        - removes its downstream ACC nodes from ACC supported set,
        - if any downstream ACC node produces non-tensor output,
          then it gets added into the worklist.
        """
    while cpu_worklist:
        node = cpu_worklist.pop(0)
        for user in node.users:
            if user in self.acc_nodes:
                self.acc_nodes.remove(user)
                if not is_node_output_tensor(user):
                    cpu_worklist.append(user)