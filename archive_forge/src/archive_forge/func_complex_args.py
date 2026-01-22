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
def complex_args(rate: StrictFloat, steps: PositiveInt=10, log_level: constr(regex='(DEBUG|INFO|WARNING|ERROR)')='ERROR'):
    return None