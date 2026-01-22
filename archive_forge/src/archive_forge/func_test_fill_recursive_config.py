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
def test_fill_recursive_config():
    valid_config = {'outer_req': 1, 'level2_req': {'hello': 4, 'world': 7}}
    filled, _, validation = my_registry._fill(valid_config, ComplexSchema)
    assert filled['outer_req'] == 1
    assert filled['outer_opt'] == 'default value'
    assert filled['level2_req']['hello'] == 4
    assert filled['level2_req']['world'] == 7
    assert filled['level2_opt']['required'] == 1
    assert filled['level2_opt']['optional'] == 'default value'