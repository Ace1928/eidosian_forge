import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
def check_node_fails(node: torch.fx.Node) -> Optional[ValidationException]:
    number = node.meta[SHAPEENV_EVENT_KEY]
    shape_env = replay_shape_env_events(events[:number + 1])
    shape_env.graph.lint()
    return check_shapeenv_fails(shape_env, events[number].tracked_fakes)