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
def test_config_dataclasses():
    cat = Cat('testcat', value_in=1, value_out=2)
    config = {'cfg': {'@cats': 'catsie.v3', 'arg': cat}}
    result = my_registry.resolve(config)['cfg']
    assert isinstance(result, Cat)
    assert result.name == cat.name
    assert result.value_in == cat.value_in
    assert result.value_out == cat.value_out