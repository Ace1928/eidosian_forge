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
def compile_fx(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor], inner_compile: Callable[..., Any]=compile_fx_inner, config_patches: Optional[Dict[str, Any]]=None, decompositions: Optional[Dict[OpOverload, Callable[..., Any]]]=None):
    """Main entrypoint to a compile given FX graph"""
    if config_patches:
        with config.patch(config_patches):
            return compile_fx(model_, example_inputs_, inner_compile=config.patch(config_patches)(inner_compile), decompositions=decompositions)
    if config.cpp_wrapper:
        with config.patch({'cpp_wrapper': False, 'triton.autotune_cublasLt': False, 'triton.cudagraphs': False, 'triton.store_cubin': True}), V.set_real_inputs(example_inputs_):
            inputs_ = example_inputs_
            if isinstance(model_, torch.fx.GraphModule):
                fake_inputs = [node.meta.get('val') for node in model_.graph.nodes if node.op == 'placeholder']
                if all((v is not None for v in fake_inputs)):
                    for idx, fi, i in zip(count(), fake_inputs, inputs_):
                        if fi.device != i.device:
                            raise ValueError(f'Device mismatch between fake input and example input at position #{idx}: {fi.device} vs {i.device}. If the model was exported via torch.export(), make sure torch.export() and torch.aot_compile() run on the same device.')
                    inputs_ = fake_inputs
            return compile_fx(model_, inputs_, inner_compile=functools.partial(inner_compile, cpp_wrapper=True), decompositions=decompositions)
    recursive_compile_fx = functools.partial(compile_fx, inner_compile=inner_compile, decompositions=decompositions)
    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(model_, example_inputs_, recursive_compile_fx)
    if isinstance(model_, torch.fx.GraphModule):
        if isinstance(model_.graph._codegen, _PyTreeCodeGen):
            return handle_dynamo_export_graph(model_, example_inputs_, recursive_compile_fx)
        model_ = pre_grad_passes(model_, example_inputs_)
    if any((isinstance(x, (list, tuple, dict)) for x in example_inputs_)):
        return flatten_graph_inputs(model_, example_inputs_, recursive_compile_fx)
    assert not config._raise_error_for_testing
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(config.triton.cudagraphs)
    forward_device = BoxedDeviceIndex(None)
    graph_id = next(_graph_counter)
    decompositions = decompositions if decompositions is not None else select_decomp_table()

    @dynamo_utils.dynamo_timed
    def fw_compiler_base(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor], is_inference: bool):
        if is_inference:
            joint_graph_passes(model)
        num_rng_seed_offset_inputs = 2 if functorch_config.functionalize_rng_ops else 0
        fixed = len(example_inputs) - num_example_inputs - num_rng_seed_offset_inputs
        user_visible_outputs = set()
        if config.keep_output_stride:
            *_, model_outputs_node = model.graph.nodes
            assert model_outputs_node.op == 'output'
            model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
            num_model_outputs = len(model_outputs)
            context = torch._guards.TracingContext.try_get()
            if context is not None and context.fw_metadata and (not is_inference):
                original_output_start_index = context.fw_metadata.num_mutated_inp_runtime_indices
            else:
                original_output_start_index = 0
            if isinstance(model_, torch.fx.GraphModule):
                *_, orig_model_outputs_node = model_.graph.nodes
                assert orig_model_outputs_node.op == 'output'
                orig_model_outputs, _ = pytree.tree_flatten(orig_model_outputs_node.args)
                num_orig_model_outputs = len(orig_model_outputs)
            else:
                num_orig_model_outputs = num_model_outputs
            assert num_orig_model_outputs <= num_model_outputs
            orig_output_end_idx = original_output_start_index + num_orig_model_outputs
            assert orig_output_end_idx <= num_model_outputs
            user_visible_outputs = {n.name for n in model_outputs[original_output_start_index:orig_output_end_idx] if isinstance(n, torch.fx.Node)}
        return inner_compile(model, example_inputs, num_fixed=fixed, cudagraphs=cudagraphs, graph_id=graph_id, is_inference=is_inference, boxed_forward_device_index=forward_device, user_visible_outputs=user_visible_outputs)
    fw_compiler = functools.partial(fw_compiler_base, is_inference=False)
    if config.freezing and (not torch.is_grad_enabled()):
        inference_compiler = functools.partial(fw_compiler_freezing, dynamo_model=model_, num_example_inputs=num_example_inputs, inner_compile=inner_compile, cudagraphs=cudagraphs, graph_id=graph_id, forward_device=forward_device)
    else:
        inference_compiler = functools.partial(fw_compiler_base, is_inference=True)

    def partition_fn(graph, joint_inputs, **kwargs):
        joint_graph_passes(graph)
        return min_cut_rematerialization_partition(graph, joint_inputs, **kwargs, compiler='inductor')

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        fixed = count_tangents(model)
        return inner_compile(model, example_inputs, num_fixed=fixed, cudagraphs=cudagraphs, is_backward=True, graph_id=graph_id, boxed_forward_device_index=forward_device)
    fake_mode = detect_fake_mode(example_inputs_) or torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
    tracing_context = torch._guards.TracingContext.try_get() or torch._guards.TracingContext(fake_mode)
    if V.aot_compilation is True:
        gm, graph_signature = aot_export_module(model_, example_inputs_, trace_joint=False, decompositions=decompositions)
        unlifted_gm = _unlift_graph(model_, gm, graph_signature)
        with V.set_fake_mode(fake_mode), compiled_autograd.disable():
            return inference_compiler(unlifted_gm, example_inputs_)
    with V.set_fake_mode(fake_mode), torch._guards.tracing(tracing_context), compiled_autograd.disable():
        return aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler, inference_compiler=inference_compiler, decompositions=decompositions, partition_fn=partition_fn, keep_inference_input_mutations=True)(model_, example_inputs_)