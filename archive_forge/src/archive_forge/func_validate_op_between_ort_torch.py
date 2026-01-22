from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]
import torch
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
@diagnostics.diagnose_call(diagnostics.rules.op_level_debugging, diagnostic_message_formatter=_op_level_debug_message_formatter)
def validate_op_between_ort_torch(diagnostic_context: diagnostics.DiagnosticContext, node: torch.fx.Node, symbolic_fn: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction], fx_args: List[fx_type_utils.Argument], fx_kwargs: Dict[str, fx_type_utils.Argument], fx_graph_module: torch.fx.GraphModule):
    """Validate the op between ONNX Runtime and PyTorch.

    The function will run the op in ONNX Runtime and PyTorch and compare the
    results. It doesn't break the exporting process, but saves each op validated
    result into SARIF, under the section of `fx_onnx_interpreter`.

    There are three signs can be found:
    1. Blue: Pass
    2. Yellow: Bypass

    Args:
        node (torch.fx.Node): The validated fx.node
        symbolic_fn (Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction]): The corresponded ONNX node
        torch_args (list): torch argument inputs
        torch_kwargs (dict): torch keyword argument inputs
        fx_graph_module (torch.fx.GraphModule): The fx.GraphModule that contains the nodes
    """
    try:
        torch_args, torch_kwargs = _wrap_fx_args_as_torch_args(fx_args, fx_kwargs, fx_graph_module)
    except ValueError as value_error:
        diagnostic = diagnostic_context.inflight_diagnostic()
        with diagnostic.log_section(logging.WARNING, 'Op level debug fails due to unsupported input types'):
            diagnostic.log_source_exception(logging.WARNING, value_error)
        diagnostic.level = diagnostics.levels.WARNING
        return
    with evaluator.default_as(evaluator.ort_evaluator):
        try:
            expected_outputs = node.target(*torch_args, **torch_kwargs)
        except IndexError as index_error:
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(logging.WARNING, 'Op level debug is bypassed'):
                diagnostic.log_source_exception(logging.WARNING, index_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        except RuntimeError as runtime_error:
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(logging.WARNING, 'Op level debug fails on PyTorch'):
                diagnostic.log_source_exception(logging.WARNING, runtime_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        try:
            function_eager_inputs, function_eager_attributes = _convert_torch_args_to_onnxfunction_args(symbolic_fn.param_schemas(), torch_args, torch_kwargs, allow_extra_kwargs=True)
            function_eager_attributes = fx_onnx_interpreter.filter_incompatible_and_dtype_convert_kwargs(function_eager_attributes)
        except TypeError as type_error:
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(logging.WARNING, 'Op level debug is bypassed'):
                diagnostic.log_source_exception(logging.WARNING, type_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        try:
            ort_outputs = symbolic_fn(*function_eager_inputs, **function_eager_attributes)
        except RuntimeError as runtime_error:
            diagnostic = diagnostic_context.inflight_diagnostic()
            with diagnostic.log_section(logging.WARNING, 'Op level debug fails on ONNXRUNTIME'):
                diagnostic.log_source_exception(logging.WARNING, runtime_error)
            diagnostic.level = diagnostics.levels.WARNING
            return
        flattened_torch_outputs, _ = _pytree.tree_flatten(expected_outputs)
        flattened_function_outputs, _ = _pytree.tree_flatten(ort_outputs)
        assert flattened_torch_outputs
        assert len(flattened_torch_outputs) == len(flattened_function_outputs)
        for torch_output, function_output in zip(flattened_torch_outputs, flattened_function_outputs):
            if isinstance(torch_output, torch.Tensor) and fx_type_utils.is_torch_complex_dtype(torch_output.dtype):
                torch_output = torch.view_as_real(torch_output.resolve_conj())
            try:
                if isinstance(function_output, onnxscript.tensor.Tensor):
                    function_output = function_output.value
                torch.testing.assert_close(torch.tensor(function_output).cpu(), torch_output.cpu() if isinstance(torch_output, torch.Tensor) else torch.tensor(torch_output).cpu(), rtol=0.0001, atol=0.001)
            except AssertionError as e:
                diagnostic = diagnostic_context.inflight_diagnostic()
                with diagnostic.log_section(logging.WARNING, 'Validation failed'):
                    diagnostic.log_source_exception(logging.WARNING, e)
                diagnostic.level = diagnostics.levels.WARNING