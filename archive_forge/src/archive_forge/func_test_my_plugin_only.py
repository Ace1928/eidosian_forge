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
def test_my_plugin_only(self):
    self.check_path(['myplugin'], ['myplugin', '-user', '-core', '-site'])