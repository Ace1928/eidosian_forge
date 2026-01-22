import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
def has_higher_order_op(gm):
    for node in gm.graph.nodes:
        if node.op == 'get_attr':
            maybe_param = getattr(gm, node.target)
            if isinstance(maybe_param, torch.fx.GraphModule):
                return True
    return False