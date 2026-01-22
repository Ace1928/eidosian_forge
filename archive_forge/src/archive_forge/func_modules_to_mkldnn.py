import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
import copy
from collections import defaultdict
import torch.utils.mkldnn as th_mkldnn
import operator
import time
import logging
from enum import Enum
def modules_to_mkldnn(nodes: List[fx.Node], modules: Dict[str, nn.Module]):
    """
    For each node, if it's a module that can be preconverted into MKLDNN,
    then we do so and create a mapping to allow us to convert from the MKLDNN
    version of the module to the original.
    """
    old_modules: Dict[nn.Module, nn.Module] = {}
    for node in nodes:
        if node.op == 'call_module':
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if type(cur_module) in mkldnn_map:
                new_module = mkldnn_map[type(cur_module)](cur_module, torch.float)
                assert isinstance(new_module, nn.Module)
                old_modules[new_module] = copy.deepcopy(cur_module)
                replace_node_module(node, modules, new_module)
    return old_modules