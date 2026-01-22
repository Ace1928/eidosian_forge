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
def get_group_from_info(group_info: TyperInfo, *, pretty_exceptions_short: bool, rich_markup_mode: MarkupMode) -> TyperGroup:
    assert group_info.typer_instance, 'A Typer instance is needed to generate a Click Group'
    commands: Dict[str, click.Command] = {}
    for command_info in group_info.typer_instance.registered_commands:
        command = get_command_from_info(command_info=command_info, pretty_exceptions_short=pretty_exceptions_short, rich_markup_mode=rich_markup_mode)
        if command.name:
            commands[command.name] = command
    for sub_group_info in group_info.typer_instance.registered_groups:
        sub_group = get_group_from_info(sub_group_info, pretty_exceptions_short=pretty_exceptions_short, rich_markup_mode=rich_markup_mode)
        if sub_group.name:
            commands[sub_group.name] = sub_group
    solved_info = solve_typer_info_defaults(group_info)
    params, convertors, context_param_name = get_params_convertors_ctx_param_name_from_function(solved_info.callback)
    cls = solved_info.cls or TyperGroup
    assert issubclass(cls, TyperGroup)
    group = cls(name=solved_info.name or '', commands=commands, invoke_without_command=solved_info.invoke_without_command, no_args_is_help=solved_info.no_args_is_help, subcommand_metavar=solved_info.subcommand_metavar, chain=solved_info.chain, result_callback=solved_info.result_callback, context_settings=solved_info.context_settings, callback=get_callback(callback=solved_info.callback, params=params, convertors=convertors, context_param_name=context_param_name, pretty_exceptions_short=pretty_exceptions_short), params=params, help=solved_info.help, epilog=solved_info.epilog, short_help=solved_info.short_help, options_metavar=solved_info.options_metavar, add_help_option=solved_info.add_help_option, hidden=solved_info.hidden, deprecated=solved_info.deprecated, rich_markup_mode=rich_markup_mode, rich_help_panel=solved_info.rich_help_panel)
    return group