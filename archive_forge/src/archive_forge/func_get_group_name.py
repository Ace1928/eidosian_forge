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
def get_group_name(typer_info: TyperInfo) -> Optional[str]:
    if typer_info.callback:
        return get_command_name(typer_info.callback.__name__)
    if typer_info.typer_instance:
        registered_callback = typer_info.typer_instance.registered_callback
        if registered_callback:
            if registered_callback.callback:
                return get_command_name(registered_callback.callback.__name__)
        if typer_info.typer_instance.info.callback:
            return get_command_name(typer_info.typer_instance.info.callback.__name__)
    return None