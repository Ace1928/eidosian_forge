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
def test_plugin_with_bad_name_does_not_load(self):
    name = 'brz-bad plugin-name..py'
    open(name, 'w').close()
    log = self.load_and_capture(name)
    self.assertContainsRe(log, "Unable to load 'brz-bad plugin-name\\.' in '.*' as a plugin because the file path isn't a valid module name; try renaming it to 'bad_plugin_name_'\\.")