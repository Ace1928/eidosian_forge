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
def test_handle_generic_type():
    """Test that validation can handle checks against arbitrary generic
    types in function argument annotations."""
    cfg = {'@cats': 'generic_cat.v1', 'cat': {'@cats': 'int_cat.v1', 'value_in': 3}}
    cat = my_registry.resolve({'test': cfg})['test']
    assert isinstance(cat, Cat)
    assert cat.value_in == 3
    assert cat.value_out is None
    assert cat.name == 'generic_cat'