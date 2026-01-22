import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
def save_config(self):
    """Convert config to a pickled blob"""
    config = dict(self._config)
    for key in config.get('_save_config_ignore', ()):
        config.pop(key)
    return pickle.dumps(config, protocol=2)