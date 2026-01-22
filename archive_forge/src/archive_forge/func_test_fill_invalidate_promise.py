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
def test_fill_invalidate_promise():
    config = {'required': 1, 'optional': {'@cats': 'catsie.v1', 'evil': False}}
    with pytest.raises(ConfigValidationError):
        my_registry._fill(config, HelloIntsSchema)
    config['optional']['whiskers'] = True
    with pytest.raises(ConfigValidationError):
        my_registry._fill(config, DefaultsSchema)