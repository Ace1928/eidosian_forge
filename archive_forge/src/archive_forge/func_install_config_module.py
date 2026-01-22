import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
def install_config_module(module):
    """
    Converts a module-level config into a `ConfigModule()`
    """

    class ConfigModuleInstance(ConfigModule):
        _bypass_keys = set()

    def visit(source, dest, prefix):
        """Walk the module structure and move everything to module._config"""
        for key, value in list(source.__dict__.items()):
            if key.startswith('__') or isinstance(value, (ModuleType, FunctionType)):
                continue
            name = f'{prefix}{key}'
            if isinstance(value, CONFIG_TYPES):
                config[name] = value
                default[name] = value
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                assert value.__module__ == module.__name__
                proxy = SubConfigProxy(module, f'{name}.')
                visit(value, proxy, f'{name}.')
                setattr(dest, key, proxy)
            else:
                raise AssertionError(f'Unhandled config {key}={value} ({type(value)})')
    config = dict()
    default = dict()
    visit(module, module, '')
    module._config = config
    module._default = default
    module._allowed_keys = set(config.keys())
    module.__class__ = ConfigModuleInstance