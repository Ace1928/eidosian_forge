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
def test_plugin_loaded(self):
    self.assertPluginUnknown('plugin')
    self.assertIs(None, breezy.plugin.get_loaded_plugin('plugin'))
    self.setup_plugin()
    p = breezy.plugin.get_loaded_plugin('plugin')
    self.assertIsInstance(p, breezy.plugin.PlugIn)
    self.assertIs(p.module, sys.modules[self.module_prefix + 'plugin'])