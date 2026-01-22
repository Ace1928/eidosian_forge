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
def test_invalidate_extra_args():
    invalid_config = {'hello': 1, 'world': 2, 'extra': 3}
    with pytest.raises(ConfigValidationError):
        my_registry._fill(invalid_config, HelloIntsSchema)