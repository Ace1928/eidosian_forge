import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
def test_compiled_loaded(self):
    self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
    self.load_with_paths(['standard'])
    self.assertTestFooLoadedFrom('non-standard-dir')
    self.assertIsSameRealPath('non-standard-dir/__init__.py', self.module.test_foo.__file__)
    os.remove('non-standard-dir/__init__.py')
    self.promote_cache('non-standard-dir')
    self.reset()
    self.load_with_paths(['standard'])
    self.assertTestFooLoadedFrom('non-standard-dir')
    suffix = plugin.COMPILED_EXT
    self.assertIsSameRealPath('non-standard-dir/__init__' + suffix, self.module.test_foo.__file__)