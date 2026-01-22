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
def test_validate_simple_config():
    simple_config = {'hello': 1, 'world': 2}
    f, _, v = my_registry._fill(simple_config, HelloIntsSchema)
    assert f == simple_config
    assert v == simple_config