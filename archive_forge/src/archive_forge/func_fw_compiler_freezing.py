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
def fw_compiler_freezing(aot_autograd_model: torch.fx.GraphModule, aot_example_inputs: List[torch.Tensor], dynamo_model: torch.fx.GraphModule, num_example_inputs: int, inner_compile: Callable[..., Any], cudagraphs: BoxedBool, graph_id: int, forward_device: BoxedDeviceIndex):
    from torch._inductor.freezing import convert_conv_weights_to_channels_last, freeze
    joint_graph_passes(aot_autograd_model)
    layout_opt = GraphLowering.decide_layout_opt(aot_autograd_model, is_inference=True)
    if layout_opt:
        fake_tensor_prop(aot_autograd_model, aot_example_inputs, True)
        convert_conv_weights_to_channels_last(aot_autograd_model)
    opt_model, preserved_arg_indices = freeze(dynamo_model, aot_autograd_model, aot_example_inputs)
    aot_example_inputs = [aot_example_inputs[ind] for ind in preserved_arg_indices]
    num_fixed = len(preserved_arg_indices) - num_example_inputs
    fake_mode = detect_fake_mode(aot_example_inputs)
    *_, model_outputs_node = opt_model.graph.nodes
    model_outputs = model_outputs_node.args[0]
    user_visible_outputs = [n.name for n in model_outputs if isinstance(n, torch.fx.Node)]
    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context is not None:
        params_flat = tracing_context.params_flat
        assert params_flat is not None
        for i in range(len(params_flat)):
            if i not in preserved_arg_indices:
                params_flat[i] = None
    with mock.patch.object(fake_mode, 'allow_non_fake_inputs', True):
        optimized_function = inner_compile(opt_model, aot_example_inputs, num_fixed=num_fixed, cudagraphs=cudagraphs, graph_id=graph_id, is_inference=True, boxed_forward_device_index=forward_device, layout_opt=layout_opt, user_visible_outputs=user_visible_outputs)
    if V.aot_compilation is True:
        return optimized_function

    def wrapper(args):
        args_new = [args[i] for i in preserved_arg_indices]
        args.clear()
        return optimized_function(args_new)
    wrapper._boxed_call = True
    return wrapper