from __future__ import annotations
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils
@_beartype.beartype
def format_function_signature_in_markdown(fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any], format_argument: Callable[[Any], str]=formatter.format_argument) -> str:
    msg_list = [f'### Function Signature {formatter.display_name(fn)}']
    state = utils.function_state(fn, args, kwargs)
    for k, v in state.items():
        msg_list.append(f'- {k}: {format_argument(v)}')
    return '\n'.join(msg_list)