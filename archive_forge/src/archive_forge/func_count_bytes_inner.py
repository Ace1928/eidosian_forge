import contextlib
import dataclasses
import functools
import logging
import os
import sys
import time
import warnings
from itertools import count
from typing import (
from unittest import mock
from functorch.compile import min_cut_rematerialization_partition
import torch._functorch.config as functorch_config
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo import (
from torch._dynamo.utils import detect_fake_mode, lazy_format_graph_code
from torch._functorch.aot_autograd import aot_export_module, make_boxed_func
from torch._inductor.codecache import code_hash, CompiledFxGraph, FxGraphCache
from torch._inductor.debug import save_args_for_compile_fx_inner
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from .._dynamo.backends.common import aot_autograd
from ..fx.graph import _PyTreeCodeGen
from . import config, metrics
from .debug import DebugContext
from .decomposition import select_decomp_table
from .fx_passes.joint_graph import joint_graph_passes
from .fx_passes.post_grad import post_grad_passes, view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes
from .graph import GraphLowering
from .ir import ExternKernelNode
from .utils import get_dtype_size, has_incompatible_cudagraph_ops
from .virtualized import V
@DebugContext.wrap
def count_bytes_inner(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], num_fixed: int=0, **kwargs):
    shape_env = _shape_env_from_inputs(example_inputs)
    fake_mode = fake_tensor_prop(gm, example_inputs)
    with V.set_fake_mode(fake_mode):
        post_grad_passes(gm, False)
    graph = GraphLowering(gm, shape_env=shape_env, num_static_inputs=num_fixed)
    with V.set_graph_handler(graph), V.set_real_inputs(example_inputs):
        graph.run(*example_inputs)
        num_bytes, nodes_num_elem, node_runtimes = graph.count_bytes()
        metrics.num_bytes_accessed += num_bytes
        metrics.nodes_num_elem += nodes_num_elem
        metrics.node_runtimes += node_runtimes
    return make_boxed_func(gm.forward)