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
def get_node_submodule_map(self) -> Dict[str, str]:
    """ Returns a map from node name to submodule name, e.g.
            node: main_module_impl_impl_over_arch_unary_multiple_embedding
              _pooling_embedding_pooling_sparse_entity_equivalence_key
              _proxy_embedding_bag
            maps to submodule name of: _run_on_acc_1
        """
    return self._node_submodule_map