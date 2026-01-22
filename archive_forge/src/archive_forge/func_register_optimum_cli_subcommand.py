import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
from ..utils import logging
from .base import BaseOptimumCLICommand, CommandInfo, RootOptimumCLICommand
from .env import EnvironmentCommand
from .export import ExportCommand
from .onnxruntime import ONNXRuntimeCommand
def register_optimum_cli_subcommand(command_or_command_info: Union[Type[BaseOptimumCLICommand], CommandInfo], parent_command: BaseOptimumCLICommand):
    """
    Registers a command as being a subcommand of `parent_command`.

    Args:
        command_or_command_info (`Union[Type[BaseOptimumCLICommand], CommandInfo]`):
            The command to register.
        parent_command (`BaseOptimumCLICommand`):
            The instance of the parent command.
    """
    if not isinstance(command_or_command_info, CommandInfo):
        command_info = CommandInfo(command_or_command_info.COMMAND.name, help=command_or_command_info.COMMAND.help, subcommand_class=command_or_command_info)
    else:
        command_info = command_or_command_info
    command_info.is_subcommand_info_or_raise()
    parent_command.register_subcommand(command_info)