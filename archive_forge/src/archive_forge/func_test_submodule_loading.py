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
def test_submodule_loading(self):
    self.create_plugin_package('test_bar', dir='non-standard-dir/test_bar')
    self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
    self.update_module_paths(['standard'])
    import breezy.testingplugins.test_foo
    self.assertEqual(self.module_prefix + 'test_foo', self.module.test_foo.__package__)
    import breezy.testingplugins.test_foo.test_bar
    self.assertIsSameRealPath('non-standard-dir/test_bar/__init__.py', self.module.test_foo.test_bar.__file__)