from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def sarif(self) -> sarif.Run:
    """Returns the SARIF Run object."""
    unique_rules = {diagnostic.rule for diagnostic in self.diagnostics}
    return sarif.Run(sarif.Tool(driver=sarif.ToolComponent(name=self.name, version=self.version, rules=[rule.sarif() for rule in unique_rules])), results=[diagnostic.sarif() for diagnostic in self.diagnostics])