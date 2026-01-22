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
def collect_str_args(e):
    if not (z3.is_app(e) and e.decl().kind() == kind):
        return [z3str(e)]
    else:
        return [x for i in range(e.num_args()) for x in collect_str_args(e.arg(i))]