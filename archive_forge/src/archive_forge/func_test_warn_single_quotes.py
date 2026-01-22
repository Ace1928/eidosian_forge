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
def test_warn_single_quotes():
    str_cfg = "\n    [project]\n    commands = 'do stuff'\n    "
    with pytest.warns(UserWarning, match='single-quoted'):
        Config().from_str(str_cfg)
    str_cfg = "\n    [project]\n    commands = some'thing\n    "
    Config().from_str(str_cfg)