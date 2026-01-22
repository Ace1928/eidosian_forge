import hashlib
import logging
import operator
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple
import sympy
import torch
import torch._logging
import torch.fx
from torch._decomp import get_decompositions
from torch._dynamo.utils import defake, dynamo_timed
from torch._logging import LazyString
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import has_free_symbols, ShapeEnv, SymTypes
from torch.utils._mode_utils import no_dispatch
from . import config, ir
from .codegen.common import (
from .codegen.wrapper import CppWrapperCodeGen, CudaWrapperCodeGen, WrapperCodeGen
from .exc import (
from .ir import (
from .lowering import (
from .sizevars import SizeVarAllocator
from .utils import convert_shape_to_inductor, gather_origins, get_sympy_Expr_dtype
from .virtualized import V
@staticmethod
def decide_layout_opt(gm, *, is_inference) -> bool:
    """
        Decide if we should enable layout optimization for this graph based on
        heuristics.
        """
    if not config.layout_optimization:
        return False
    if config.force_layout_optimization:
        return True
    conv_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.convolution.default]
    nconv = len(conv_nodes)
    if nconv == 0:
        return False
    if torch.version.hip and torch.cuda.is_available():
        return False
    if all((n.args[idx].meta['val'].device == torch.device('cpu') for n in conv_nodes for idx in [0, 1])) and torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
        return True
    if len(list(gm.graph.nodes)) >= 300 * nconv:
        log.debug('Skipped layout opt because only a few conv')
        return False
    if any((has_free_symbols(n.args[idx].meta['val']) for n in conv_nodes for idx in [0, 1])):
        log.debug('See perf regression with dynamic shape. Follow up in https://github.com/pytorch/pytorch/issues/102670')
        return False

    def is_grouped(n):
        return n.args[-1] > 1 and n.args[1].meta['val'].size(1) > 1

    def is_in_out_channel(n):
        return n.args[1].meta['val'].size(0) * 2 <= n.args[1].meta['val'].size(1) and n.args[1].meta['val'].size(2) > 1

    def is_small_channel(n):
        return n.args[1].meta['val'].size(0) <= 64 and n.args[1].meta['val'].size(1) <= 64
    if is_inference:
        from torch.utils.flop_counter import FlopCounterMode
        flop_counts: Dict[str, float] = defaultdict(float)
        for node in conv_nodes:
            success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(node)
            if success:
                with FlopCounterMode(display=False) as flop_counter_mode:
                    with V.fake_mode:
                        node.target(*args, **kwargs)
                counted_flops = flop_counter_mode.get_total_flops()
                if is_grouped(node):
                    node_type = 'grouped'
                elif is_small_channel(node):
                    node_type = 'small'
                elif is_in_out_channel(node):
                    node_type = 'in_out'
                else:
                    node_type = 'default'
                flop_counts[node_type] += counted_flops
            else:
                log.debug('Conv inputs meta not found')
        GROUPED_MULTIPLIER = 1.358
        DEFAULT_MULTIPLIER = 0.823
        IN_OUT_MULTIPLIER = 0.725
        SMALL_MULTIPLIER = 0.783
        total_flops = sum(flop_counts.values())
        weighted_flops = flop_counts['grouped'] * GROUPED_MULTIPLIER + flop_counts['small'] * SMALL_MULTIPLIER + flop_counts['in_out'] * IN_OUT_MULTIPLIER + flop_counts['default'] * DEFAULT_MULTIPLIER
        do_layout_opt = weighted_flops <= total_flops
        if not do_layout_opt:
            log.debug('Skipped layout opt in inference because weighted flops indicate slowdown, default: %d, channels last: %d', total_flops, weighted_flops)
        return do_layout_opt
    if any((is_grouped(n) for n in conv_nodes)):
        log.debug('Skip layout opt because found grouped convolution with >1 in_channels!')
        return False
    if any((is_in_out_channel(n) for n in conv_nodes)):
        log.debug('Skip layout opt because some convolutions have smaller out_channel')
        return False
    if all((is_small_channel(n) for n in conv_nodes)):
        log.debug('Skip layout opt because all convolution channels are too small')
        return False
    return True