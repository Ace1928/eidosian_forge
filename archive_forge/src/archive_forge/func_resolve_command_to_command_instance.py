import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
from ..utils import logging
from .base import BaseOptimumCLICommand, CommandInfo, RootOptimumCLICommand
from .env import EnvironmentCommand
from .export import ExportCommand
from .onnxruntime import ONNXRuntimeCommand
def resolve_command_to_command_instance(root: RootOptimumCLICommand, commands: List[Type[BaseOptimumCLICommand]]) -> Dict[Type[BaseOptimumCLICommand], BaseOptimumCLICommand]:
    """
    Retrieves command instances in the root CLI command from a list of command classes.

    Args:
        root (`RootOptimumCLICommand`):
            The root CLI command instance.
        commands (`List[Type[BaseOptimumCLICommand]]`):
            The list of command classes to retrieve the instances of in root.

    Returns:
        `Dict[Type[BaseOptimumCLICommand], BaseOptimumCLICommand]`: A dictionary mapping a command class to a command
        instance in the root CLI.
    """
    to_visit = [root]
    remaining_commands = set(commands)
    command2command_instance = {}
    while to_visit:
        current_command_instance = to_visit.pop(0)
        if current_command_instance.__class__ in remaining_commands:
            remaining_commands.remove(current_command_instance.__class__)
            command2command_instance[current_command_instance.__class__] = current_command_instance
        if not remaining_commands:
            break
        to_visit += current_command_instance.registered_subcommands
    if remaining_commands:
        class_names = (command.__name__ for command in remaining_commands)
        raise RuntimeError(f'Could not find an instance of the following commands in the CLI {root}: {', '.join(class_names)}.')
    return command2command_instance