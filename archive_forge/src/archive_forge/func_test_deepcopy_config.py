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
def test_deepcopy_config():
    config = Config({'a': 1, 'b': {'c': 2, 'd': 3}})
    copied = config.copy()
    assert config == copied
    assert config is not copied