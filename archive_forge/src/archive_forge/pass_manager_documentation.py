import inspect
import logging
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult

        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.
        