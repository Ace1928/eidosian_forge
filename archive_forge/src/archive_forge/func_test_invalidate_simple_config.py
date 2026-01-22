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
def test_invalidate_simple_config():
    invalid_config = {'hello': 1, 'world': 'hi!'}
    with pytest.raises(ConfigValidationError) as exc_info:
        my_registry._fill(invalid_config, HelloIntsSchema)
    error = exc_info.value
    assert len(error.errors) == 1
    assert 'type_error.integer' in error.error_types