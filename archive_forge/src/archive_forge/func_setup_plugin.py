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
def setup_plugin(self, source=''):
    self.assertPluginUnknown('plugin')
    with open('plugin.py', 'w') as f:
        f.write(source + '\n')
    self.load_with_paths(['.'])