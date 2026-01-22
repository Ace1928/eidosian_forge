import inspect
import os
import sys
import traceback
from datetime import datetime
from enum import Enum
from functools import update_wrapper
from pathlib import Path
from traceback import FrameSummary, StackSummary
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from uuid import UUID
import click
from .completion import get_completion_inspect_parameters
from .core import MarkupMode, TyperArgument, TyperCommand, TyperGroup, TyperOption
from .models import (
from .utils import get_params_from_function
def except_hook(exc_type: Type[BaseException], exc_value: BaseException, tb: Optional[TracebackType]) -> None:
    exception_config: Union[DeveloperExceptionConfig, None] = getattr(exc_value, _typer_developer_exception_attr_name, None)
    standard_traceback = os.getenv('_TYPER_STANDARD_TRACEBACK')
    if standard_traceback or not exception_config or (not exception_config.pretty_exceptions_enable):
        _original_except_hook(exc_type, exc_value, tb)
        return
    typer_path = os.path.dirname(__file__)
    click_path = os.path.dirname(click.__file__)
    supress_internal_dir_names = [typer_path, click_path]
    exc = exc_value
    if rich:
        rich_tb = Traceback.from_exception(type(exc), exc, exc.__traceback__, show_locals=exception_config.pretty_exceptions_show_locals, suppress=supress_internal_dir_names)
        console_stderr.print(rich_tb)
        return
    tb_exc = traceback.TracebackException.from_exception(exc)
    stack: List[FrameSummary] = []
    for frame in tb_exc.stack:
        if any((frame.filename.startswith(path) for path in supress_internal_dir_names)):
            if not exception_config.pretty_exceptions_short:
                stack.append(traceback.FrameSummary(filename=frame.filename, lineno=frame.lineno, name=frame.name, line=''))
        else:
            stack.append(frame)
    final_stack_summary = StackSummary.from_list(stack)
    tb_exc.stack = final_stack_summary
    for line in tb_exc.format():
        print(line, file=sys.stderr)
    return