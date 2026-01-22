import inspect
import pickle
import platform
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import catalogue
import pytest
from confection import Config, ConfigValidationError
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial
def test_config_to_str_order():
    """Test that Config.to_str orders the sections."""
    config = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}, 'f': {'g': {'h': {'i': 4, 'j': 5}}}}
    expected = '[a]\ne = 3\n\n[a.b]\nc = 1\nd = 2\n\n[f]\n\n[f.g]\n\n[f.g.h]\ni = 4\nj = 5'
    config = Config(config)
    assert config.to_str() == expected