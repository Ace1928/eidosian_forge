from __future__ import annotations
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.onnx._internal.diagnostics import infra
@_beartype.beartype
def pre_export_passes(self, options: exporter.ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
    from torch.onnx._internal.fx import analysis, passes
    diagnostic_context = options.diagnostic_context
    fx_module = passes.InsertTypePromotion(diagnostic_context, fx_module).run()
    analysis.UnsupportedFxNodesAnalysis(diagnostic_context, fx_module, options.onnxfunction_dispatcher).analyze(infra.levels.ERROR)
    return fx_module