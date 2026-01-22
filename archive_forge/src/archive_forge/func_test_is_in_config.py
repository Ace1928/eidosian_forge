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
@pytest.mark.parametrize('prop,expected', [('a.b.c', True), ('a.b', True), ('a', True), ('a.e', True), ('a.b.c.d', False)])
def test_is_in_config(prop, expected):
    config = {'a': {'b': {'c': 5, 'd': 6}, 'e': [1, 2]}}
    assert my_registry._is_in_config(prop, config) is expected