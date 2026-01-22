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
def check_path(self, expected_dirs, setting_dirs):
    if setting_dirs is None:
        del os.environ['BRZ_PLUGIN_PATH']
    else:
        os.environ['BRZ_PLUGIN_PATH'] = os.pathsep.join(setting_dirs)
    actual = [p if t == 'path' else t.upper() for p, t in plugin._env_plugin_path()]
    self.assertEqual(expected_dirs, actual)