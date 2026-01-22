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
def test_plugin_get_path_py_not_pyc(self):
    self.setup_plugin()
    self.promote_cache(self.test_dir)
    self.reset()
    self.load_with_paths(['.'])
    p = plugin.plugins()['plugin']
    plugin_path = self.test_dir + '/plugin.py'
    self.assertIsSameRealPath(plugin_path, osutils.normpath(p.path()))