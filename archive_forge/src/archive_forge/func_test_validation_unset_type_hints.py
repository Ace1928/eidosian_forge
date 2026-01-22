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
def test_validation_unset_type_hints():
    """Test that unset type hints are handled correctly (and treated as Any)."""

    @my_registry.optimizers('test_optimizer.v2')
    def test_optimizer_v2(rate, steps: int=10) -> None:
        return None
    config = {'test': {'@optimizers': 'test_optimizer.v2', 'rate': 0.1, 'steps': 20}}
    my_registry.resolve(config)