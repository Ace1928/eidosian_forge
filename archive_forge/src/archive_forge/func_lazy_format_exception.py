from __future__ import annotations
import dataclasses
import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from torch._logging import LazyString
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import sarif
def lazy_format_exception(exception: Exception) -> LazyString:
    return LazyString(lambda: '\n'.join(('```', *traceback.format_exception(type(exception), exception, exception.__traceback__), '```')))