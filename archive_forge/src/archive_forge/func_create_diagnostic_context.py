from __future__ import annotations
import contextlib
import gzip
from collections.abc import Generator
from typing import List, Optional
import torch
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
from torch.utils import cpp_backtrace
def create_diagnostic_context(self, name: str, version: str, options: Optional[infra.DiagnosticOptions]=None) -> infra.DiagnosticContext:
    """Creates a new diagnostic context.

        Args:
            name: The subject name for the diagnostic context.
            version: The subject version for the diagnostic context.
            options: The options for the diagnostic context.

        Returns:
            A new diagnostic context.
        """
    if options is None:
        options = infra.DiagnosticOptions()
    context: infra.DiagnosticContext[infra.Diagnostic] = infra.DiagnosticContext(name, version, options)
    self.contexts.append(context)
    return context