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
def test_config_fill_without_resolve():

    class BaseSchema(BaseModel):
        catsie: int
    config = {'catsie': {'@cats': 'catsie.v1', 'evil': False}}
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved['catsie'] == 'meow'
    assert filled['catsie']['cute'] is True
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(config, schema=BaseSchema)
    filled2 = my_registry.fill(config, schema=BaseSchema)
    assert filled2['catsie']['cute'] is True
    resolved = my_registry.resolve(filled2)
    assert resolved['catsie'] == 'meow'

    class BaseSchema2(BaseModel):
        catsie: Any
        other: int = 12
    config = {'catsie': {'@cats': 'dog', 'evil': False}}
    filled3 = my_registry.fill(config, schema=BaseSchema2)
    assert filled3['catsie'] == config['catsie']
    assert filled3['other'] == 12