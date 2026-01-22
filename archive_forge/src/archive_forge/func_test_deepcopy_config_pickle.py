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
@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='copy does not fail for pypy')
def test_deepcopy_config_pickle():
    numpy = pytest.importorskip('numpy')
    config = Config({'a': 1, 'b': numpy})
    with pytest.raises(ValueError):
        config.copy()