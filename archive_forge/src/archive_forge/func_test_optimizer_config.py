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
def test_optimizer_config():
    cfg = Config().from_str(OPTIMIZER_CFG)
    optimizer = my_registry.resolve(cfg, validate=True)['optimizer']
    assert optimizer.beta1 == 0.9