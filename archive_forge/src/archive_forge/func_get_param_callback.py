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
def get_param_callback(*, callback: Optional[Callable[..., Any]]=None, convertor: Optional[Callable[..., Any]]=None) -> Optional[Callable[..., Any]]:
    if not callback:
        return None
    parameters = get_params_from_function(callback)
    ctx_name = None
    click_param_name = None
    value_name = None
    untyped_names: List[str] = []
    for param_name, param_sig in parameters.items():
        if lenient_issubclass(param_sig.annotation, click.Context):
            ctx_name = param_name
        elif lenient_issubclass(param_sig.annotation, click.Parameter):
            click_param_name = param_name
        else:
            untyped_names.append(param_name)
    if untyped_names:
        value_name = untyped_names.pop()
    if untyped_names:
        if ctx_name is None:
            ctx_name = untyped_names.pop(0)
        if click_param_name is None:
            if untyped_names:
                click_param_name = untyped_names.pop(0)
        if untyped_names:
            raise click.ClickException('Too many CLI parameter callback function parameters')

    def wrapper(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
        use_params: Dict[str, Any] = {}
        if ctx_name:
            use_params[ctx_name] = ctx
        if click_param_name:
            use_params[click_param_name] = param
        if value_name:
            if convertor:
                use_value = convertor(value)
            else:
                use_value = value
            use_params[value_name] = use_value
        return callback(**use_params)
    update_wrapper(wrapper, callback)
    return wrapper