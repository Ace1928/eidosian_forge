from __future__ import annotations
import functools
import inspect
import traceback
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import _infra, formatter
@_beartype.beartype
def function_location(fn: Callable) -> _infra.Location:
    """Returns a Location for the given function."""
    source_lines, lineno, uri = _function_source_info(fn)
    snippet = source_lines[0].strip() if len(source_lines) > 0 else '<unknown>'
    return _infra.Location(uri=uri, line=lineno, snippet=snippet, message=formatter.display_name(fn))