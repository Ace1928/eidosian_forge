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
def test_config_from_str_invalid_section():
    config_str = '[a]\nb = null\n\n[a.b]\nc = 1'
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)
    config_str = '[a]\nb = null\n\n[a.b.c]\nd = 1'
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)