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
def test_positional_args_to_from_string():
    cfg = '[a]\nb = 1\n* = ["foo","bar"]'
    assert Config().from_str(cfg).to_str() == cfg
    cfg = '[a]\nb = 1\n\n[a.*.bar]\ntest = 2\n\n[a.*.foo]\ntest = 1'
    assert Config().from_str(cfg).to_str() == cfg

    @my_registry.cats('catsie.v666')
    def catsie_666(*args, meow=False):
        return args
    cfg = '[a]\n@cats = "catsie.v666"\n* = ["foo","bar"]'
    filled = my_registry.fill(Config().from_str(cfg)).to_str()
    assert filled == '[a]\n@cats = "catsie.v666"\n* = ["foo","bar"]\nmeow = false'
    resolved = my_registry.resolve(Config().from_str(cfg))
    assert resolved == {'a': ('foo', 'bar')}
    cfg = '[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\nx = 1'
    filled = my_registry.fill(Config().from_str(cfg)).to_str()
    assert filled == '[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\nx = 1'
    resolved = my_registry.resolve(Config().from_str(cfg))
    assert resolved == {'a': ({'x': 1},)}

    @my_registry.cats('catsie.v777')
    def catsie_777(y: int=1):
        return 'meow' * y
    cfg = '[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777"'
    filled = my_registry.fill(Config().from_str(cfg)).to_str()
    expected = '[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 1'
    assert filled == expected
    cfg = '[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 3'
    result = my_registry.resolve(Config().from_str(cfg))
    assert result == {'a': ('meowmeowmeow',)}