from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
@_beartype.beartype
def dynamo_export(model: Union[torch.nn.Module, Callable, torch_export.ExportedProgram], /, *model_args, export_options: Optional[ExportOptions]=None, **model_kwargs) -> ONNXProgram:
    """Export a torch.nn.Module to an ONNX graph.

    Args:
        model: The PyTorch model to be exported to ONNX.
        model_args: Positional inputs to ``model``.
        model_kwargs: Keyword inputs to ``model``.
        export_options: Options to influence the export to ONNX.

    Returns:
        An in-memory representation of the exported ONNX model.

    **Example 1 - Simplest export**
    ::

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
            def forward(self, x, bias=None):
                out = self.linear(x)
                out = out + bias
                return out
        model = MyModel()
        kwargs = {"bias": 3.}
        args = (torch.randn(2, 2, 2),)
        onnx_program = torch.onnx.dynamo_export(
            model,
            *args,
            **kwargs).save("my_simple_model.onnx")

    **Example 2 - Exporting with dynamic shapes**
    ::

        # The previous model can be exported with dynamic shapes
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_program = torch.onnx.dynamo_export(
            model,
            *args,
            **kwargs,
            export_options=export_options)
        onnx_program.save("my_dynamic_model.onnx")


    By printing input dynamic dimensions we can see the input shape is no longer (2,2,2)
    ::

        >>> print(onnx_program.model_proto.graph.input[0])
        name: "arg0"
        type {
          tensor_type {
            elem_type: 1
            shape {
              dim {
                dim_param: "arg0_dim_0"
              }
              dim {
                dim_param: "arg0_dim_1"
              }
              dim {
                dim_param: "arg0_dim_2"
              }
            }
          }
        }
    """
    if export_options is not None:
        resolved_export_options = export_options if isinstance(export_options, ResolvedExportOptions) else ResolvedExportOptions(export_options, model=model)
    else:
        resolved_export_options = ResolvedExportOptions(ExportOptions(), model=model)
    _assert_dependencies(resolved_export_options)
    try:
        return Exporter(options=resolved_export_options, model=model, model_args=model_args, model_kwargs=model_kwargs).export()
    except Exception as e:
        sarif_report_path = _DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH
        resolved_export_options.diagnostic_context.dump(sarif_report_path)
        message = f"Failed to export the model to ONNX. Generating SARIF report at '{sarif_report_path}'. SARIF is a standard format for the output of static analysis tools. SARIF logs can be loaded in VS Code SARIF viewer extension, or SARIF web viewer (https://microsoft.github.io/sarif-web-component/). Please report a bug on PyTorch Github: {_PYTORCH_GITHUB_ISSUES_URL}"
        raise OnnxExporterError(ONNXProgram._from_failure(e, resolved_export_options.diagnostic_context), message) from e