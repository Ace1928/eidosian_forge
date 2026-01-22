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
def test_make_config_positional_args():

    @my_registry.cats('catsie.v567')
    def catsie_567(*args: Optional[str], foo: str='bar'):
        assert args[0] == '^_^'
        assert args[1] == '^(*.*)^'
        assert foo == 'baz'
        return args[0]
    args = ['^_^', '^(*.*)^']
    cfg = {'config': {'@cats': 'catsie.v567', 'foo': 'baz', '*': args}}
    assert my_registry.resolve(cfg)['config'] == '^_^'