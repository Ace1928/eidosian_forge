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
def test_validation_no_validate():
    config = {'one': 1, 'two': {'three': {'@cats': 'catsie.v1', 'evil': 'false'}}}
    result = my_registry.resolve({'cfg': config}, validate=False)
    filled = my_registry.fill({'cfg': config}, validate=False)
    assert result['cfg']['one'] == 1
    assert result['cfg']['two'] == {'three': 'scratch!'}
    assert filled['cfg']['two']['three']['evil'] == 'false'
    assert filled['cfg']['two']['three']['cute'] is True