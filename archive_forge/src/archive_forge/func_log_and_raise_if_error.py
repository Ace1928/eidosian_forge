from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def log_and_raise_if_error(self, diagnostic: _Diagnostic) -> None:
    """Logs a diagnostic and raises an exception if it is an error.

        Use this method for logging non inflight diagnostics where diagnostic level is not known or
        lower than ERROR. If it is always expected raise, use `log` and explicit
        `raise` instead. Otherwise there is no way to convey the message that it always
        raises to Python intellisense and type checking tools.

        This method should be used only after all the necessary information for the diagnostic
        has been collected.

        Args:
            diagnostic: The diagnostic to add.
        """
    self.log(diagnostic)
    if diagnostic.level == infra.Level.ERROR:
        if diagnostic.source_exception is not None:
            raise diagnostic.source_exception
        raise RuntimeErrorWithDiagnostic(diagnostic)