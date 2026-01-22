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
def test_config_from_str_overrides():
    config_str = '[a]\nb = 1\n\n[a.c]\nd = 2\ne = 3\n\n[f]\ng = {"x": "y"}'
    overrides = {'a.b': 10, 'a.c.d': 20}
    config = Config().from_str(config_str, overrides=overrides)
    assert config['a']['b'] == 10
    assert config['a']['c']['d'] == 20
    assert config['a']['c']['e'] == 3
    config = Config().from_str(config_str, overrides={'a.c.f': 100})
    assert config['a']['c']['d'] == 2
    assert config['a']['c']['e'] == 3
    assert config['a']['c']['f'] == 100
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str, overrides={'f': 10})
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str, overrides={'f.g.x': 'z'})
    config_str = '[a]\nb = 1\n\n[a.c]\nd = 2\ne = ${a:b}'
    config = Config().from_str(config_str, overrides={'a.b': 10})
    assert config['a']['b'] == 10
    assert config['a']['c']['e'] == 10
    config_str = '[a]\nb = 1\n\n[a.c]\nd = 2\n[e]\nf = ${a.c}'
    config = Config().from_str(config_str, overrides={'a.c.d': 20})
    assert config['a']['c']['d'] == 20
    assert config['e']['f'] == {'d': 20}