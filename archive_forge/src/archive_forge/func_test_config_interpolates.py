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
@pytest.mark.parametrize('greeting,value,expected', [[342, '${vars.a}', int], ['342', '${vars.a}', str], ['everyone', '${vars.a}', str]])
def test_config_interpolates(greeting, value, expected):
    str_cfg = f'\n    [project]\n    my_par = {value}\n\n    [vars]\n    a = "something"\n    '
    overrides = {'vars.a': greeting}
    cfg = Config().from_str(str_cfg, overrides=overrides)
    assert type(cfg['project']['my_par']) == expected