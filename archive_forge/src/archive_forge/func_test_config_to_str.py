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
def test_config_to_str():
    cfg = Config().from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()
    cfg = Config({'optimizer': {'foo': 'bar'}}).from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()