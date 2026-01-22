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
def load_with_paths(self, paths, warn_load_problems=True):
    self.log('loading plugins!')
    plugin.load_plugins(self.update_module_paths(paths), state=self, warn_load_problems=warn_load_problems)