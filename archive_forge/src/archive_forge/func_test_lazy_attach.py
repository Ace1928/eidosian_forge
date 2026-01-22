import importlib
import os
import subprocess
import sys
import types
from unittest import mock
import pytest
import lazy_loader as lazy
from . import rank
from ._gaussian import gaussian
from .edges import sobel, scharr, prewitt, roberts
def test_lazy_attach():
    name = 'mymod'
    submods = ['mysubmodule', 'anothersubmodule']
    myall = {'not_real_submod': ['some_var_or_func']}
    locls = {'attach': lazy.attach, 'name': name, 'submods': submods, 'myall': myall}
    s = '__getattr__, __lazy_dir__, __all__ = attach(name, submods, myall)'
    exec(s, {}, locls)
    expected = {'attach': lazy.attach, 'name': name, 'submods': submods, 'myall': myall, '__getattr__': None, '__lazy_dir__': None, '__all__': None}
    assert locls.keys() == expected.keys()
    for k, v in expected.items():
        if v is not None:
            assert locls[k] == v