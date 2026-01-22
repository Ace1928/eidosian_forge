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
def test_plugin_appears_in_plugins(self):
    self.setup_plugin()
    self.assertPluginKnown('plugin')
    p = self.plugins['plugin']
    self.assertIsInstance(p, breezy.plugin.PlugIn)
    self.assertIs(p.module, sys.modules[self.module_prefix + 'plugin'])