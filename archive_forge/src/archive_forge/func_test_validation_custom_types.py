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
def test_validation_custom_types():

    def complex_args(rate: StrictFloat, steps: PositiveInt=10, log_level: constr(regex='(DEBUG|INFO|WARNING|ERROR)')='ERROR'):
        return None
    my_registry.complex = catalogue.create(my_registry.namespace, 'complex', entry_points=False)
    my_registry.complex('complex.v1')(complex_args)
    cfg = {'@complex': 'complex.v1', 'rate': 1.0, 'steps': 20, 'log_level': 'INFO'}
    my_registry.resolve({'config': cfg})
    cfg = {'@complex': 'complex.v1', 'rate': 1.0, 'steps': -1, 'log_level': 'INFO'}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve({'config': cfg})
    cfg = {'@complex': 'complex.v1', 'rate': 1.0, 'steps': 20, 'log_level': 'none'}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve({'config': cfg})
    cfg = {'@complex': 'complex.v1', 'rate': 1.0, 'steps': 20, 'log_level': 'INFO'}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(cfg)
    with pytest.raises(ConfigValidationError):
        my_registry.fill(cfg)
    cfg = {'@complex': 'complex.v1', 'rate': 1.0, '@cats': 'catsie.v1'}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve({'config': cfg})