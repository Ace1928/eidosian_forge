from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def register_argparse_argument_parameter(param_name: str, param_type: Optional[Type[Any]]) -> None:
    """
    Registers a custom argparse argument parameter.

    The registered name will then be a recognized keyword parameter to the parser's `add_argument()` function.

    An accessor functions will be added to the parameter's Action object in the form of: ``get_{param_name}()``
    and ``set_{param_name}(value)``.

    :param param_name: Name of the parameter to add.
    :param param_type: Type of the parameter to add.
    """
    attr_name = f'{_CUSTOM_ATTRIB_PFX}{param_name}'
    if param_name in CUSTOM_ACTION_ATTRIBS or hasattr(argparse.Action, attr_name):
        raise KeyError(f'Custom parameter {param_name} already exists')
    if not re.search('^[A-Za-z_][A-Za-z0-9_]*$', param_name):
        raise KeyError(f'Invalid parameter name {param_name} - cannot be used as a python identifier')
    getter_name = f'get_{param_name}'

    def _action_get_custom_parameter(self: argparse.Action) -> Any:
        f'\n        Get the custom {param_name} attribute of an argparse Action.\n\n        This function is added by cmd2 as a method called ``{getter_name}()`` to ``argparse.Action`` class.\n\n        To call: ``action.{getter_name}()``\n\n        :param self: argparse Action being queried\n        :return: The value of {param_name} or None if attribute does not exist\n        '
        return getattr(self, attr_name, None)
    setattr(argparse.Action, getter_name, _action_get_custom_parameter)
    setter_name = f'set_{param_name}'

    def _action_set_custom_parameter(self: argparse.Action, value: Any) -> None:
        f'\n        Set the custom {param_name} attribute of an argparse Action.\n\n        This function is added by cmd2 as a method called ``{setter_name}()`` to ``argparse.Action`` class.\n\n        To call: ``action.{setter_name}({param_name})``\n\n        :param self: argparse Action being updated\n        :param value: value being assigned\n        '
        if param_type and (not isinstance(value, param_type)):
            raise TypeError(f'{param_name} must be of type {param_type}, got: {value} ({type(value)})')
        setattr(self, attr_name, value)
    setattr(argparse.Action, setter_name, _action_set_custom_parameter)
    CUSTOM_ACTION_ATTRIBS.add(param_name)