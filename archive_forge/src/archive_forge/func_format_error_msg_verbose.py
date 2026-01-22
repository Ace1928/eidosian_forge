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
def format_error_msg_verbose(exc: Exception, code, record_filename=None, frame=None) -> str:
    msg = f"WON'T CONVERT {code.co_name} {code.co_filename} line {code.co_firstlineno}\n"
    msg += '=' * 10 + ' TorchDynamo Stack Trace ' + '=' * 10 + '\n'
    msg += format_exc()
    real_stack = get_real_stack(exc, frame)
    if real_stack is not None:
        msg += '\n' + '=' * 10 + ' The above exception occurred while processing the following code ' + '=' * 10 + '\n\n'
        msg += ''.join(format_list(real_stack))
        msg += '\n'
        msg += '=' * 10
    return msg