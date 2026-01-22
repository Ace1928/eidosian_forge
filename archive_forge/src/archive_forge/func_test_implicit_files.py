import textwrap
import unittest
import warnings
import importlib
import contextlib
import importlib_resources as resources
from ..abc import Traversable
from . import data01
from . import util
from . import _path
from .compat.py39 import os_helper
from .compat.py312 import import_helper
def test_implicit_files(self):
    """
        Without any parameter, files() will infer the location as the caller.
        """
    spec = {'somepkg': {'__init__.py': textwrap.dedent("\n                    import importlib_resources as res\n                    val = res.files().joinpath('res.txt').read_text(encoding='utf-8')\n                    "), 'res.txt': 'resources are the best'}}
    _path.build(spec, self.site_dir)
    assert importlib.import_module('somepkg').val == 'resources are the best'