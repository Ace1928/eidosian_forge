import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import cast, NoReturn, Optional
import torch._guards
from . import config
from .config import is_fbcode
from .utils import counters
import logging
def filter_stack(stack):
    user_stack = []
    for frame in stack:
        if 'convert_frame' in frame.filename:
            break
        if 'eval_frame' in frame.filename or 'torch._dynamo.optimize(' in frame.line:
            continue
        user_stack.append(frame)
    return user_stack