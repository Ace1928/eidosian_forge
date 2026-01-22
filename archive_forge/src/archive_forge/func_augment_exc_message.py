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
def augment_exc_message(exc: Exception, msg: str='\n', export: bool=False) -> None:
    import traceback
    real_stack = get_real_stack(exc)
    if real_stack is not None:
        msg += f'\nfrom user code:\n {''.join(traceback.format_list(real_stack))}'
    if config.replay_record_enabled and hasattr(exc, 'record_filename'):
        msg += f"\nLast frame execution written to {exc.record_filename}. To run only this frame while debugging, run torch._dynamo.replay('{exc.record_filename}').\n"
    if not config.verbose and hasattr(exc, 'real_stack'):
        msg += '\nSet TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information\n'
    if hasattr(exc, 'inner_exception') and hasattr(exc.inner_exception, 'minifier_path'):
        if hasattr(exc.inner_exception, 'buck_command'):
            msg += f'\nMinifier script written to {exc.inner_exception.minifier_path}. Run this buck command to find the smallest traced graph which reproduces this error: {exc.inner_exception.buck_command}\n'
        else:
            msg += f'\nMinifier script written to {exc.inner_exception.minifier_path}. Run this script to find the smallest traced graph which reproduces this error.\n'
    if not config.suppress_errors and (not export):
        msg += '\n\nYou can suppress this exception and fall back to eager by setting:\n    import torch._dynamo\n    torch._dynamo.config.suppress_errors = True\n'
    old_msg = '' if len(exc.args) == 0 else str(exc.args[0])
    if isinstance(exc, KeyError):
        exc.args = (KeyErrorMsg(old_msg + msg),) + exc.args[1:]
    else:
        new_msg = old_msg + msg
        exc.args = (new_msg,) + exc.args[1:]