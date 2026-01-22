import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def register_command_set(self, cmdset: CommandSet) -> None:
    """
        Installs a CommandSet, loading all commands defined in the CommandSet

        :param cmdset: CommandSet to load
        """
    existing_commandset_types = [type(command_set) for command_set in self._installed_command_sets]
    if type(cmdset) in existing_commandset_types:
        raise CommandSetRegistrationError('CommandSet ' + type(cmdset).__name__ + ' is already installed')
    all_settables = self.settables
    if self.always_prefix_settables:
        if not cmdset.settable_prefix.strip():
            raise CommandSetRegistrationError('CommandSet settable prefix must not be empty')
        for key in cmdset.settables.keys():
            prefixed_name = f'{cmdset.settable_prefix}.{key}'
            if prefixed_name in all_settables:
                raise CommandSetRegistrationError(f'Duplicate settable: {key}')
    else:
        for key in cmdset.settables.keys():
            if key in all_settables:
                raise CommandSetRegistrationError(f'Duplicate settable {key} is already registered')
    cmdset.on_register(self)
    methods = inspect.getmembers(cmdset, predicate=lambda meth: isinstance(meth, Callable) and hasattr(meth, '__name__') and meth.__name__.startswith(COMMAND_FUNC_PREFIX))
    default_category = getattr(cmdset, CLASS_ATTR_DEFAULT_HELP_CATEGORY, None)
    installed_attributes = []
    try:
        for method_name, method in methods:
            command = method_name[len(COMMAND_FUNC_PREFIX):]
            self._install_command_function(command, method, type(cmdset).__name__)
            installed_attributes.append(method_name)
            completer_func_name = COMPLETER_FUNC_PREFIX + command
            cmd_completer = getattr(cmdset, completer_func_name, None)
            if cmd_completer is not None:
                self._install_completer_function(command, cmd_completer)
                installed_attributes.append(completer_func_name)
            help_func_name = HELP_FUNC_PREFIX + command
            cmd_help = getattr(cmdset, help_func_name, None)
            if cmd_help is not None:
                self._install_help_function(command, cmd_help)
                installed_attributes.append(help_func_name)
            self._cmd_to_command_sets[command] = cmdset
            if default_category and (not hasattr(method, constants.CMD_ATTR_HELP_CATEGORY)):
                utils.categorize(method, default_category)
        self._installed_command_sets.add(cmdset)
        self._register_subcommands(cmdset)
        cmdset.on_registered()
    except Exception:
        cmdset.on_unregister()
        for attrib in installed_attributes:
            delattr(self, attrib)
        if cmdset in self._installed_command_sets:
            self._installed_command_sets.remove(cmdset)
        if cmdset in self._cmd_to_command_sets.values():
            self._cmd_to_command_sets = {key: val for key, val in self._cmd_to_command_sets.items() if val is not cmdset}
        cmdset.on_unregistered()
        raise