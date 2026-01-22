import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def print_all_graph_modules(self) -> None:
    logger.info('Printing the three fx gm:')
    logger.info('1. Setup fx.GraphModule:')
    logger.info('%s', self.setup_gm.print_readable(False))
    logger.info('2. Main fx.GraphModule:')
    logger.info('%s', self.main_gm.print_readable(False))
    logger.info('3. Cleanup fx.GraphModule:')
    logger.info('%s', self.cleanup_gm.print_readable(False))