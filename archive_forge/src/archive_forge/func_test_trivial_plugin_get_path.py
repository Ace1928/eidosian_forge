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
def test_trivial_plugin_get_path(self):
    self.setup_plugin()
    p = self.plugins['plugin']
    plugin_path = self.test_dir + '/plugin.py'
    self.assertIsSameRealPath(plugin_path, osutils.normpath(p.path()))