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
def test_load_plugin_tests_gives_load_plugin_tests_result(self):
    source = "\ndef load_tests(loader, standard_tests, pattern):\n    return 'foo'"
    self.setup_plugin(source)
    loader = tests.TestUtil.TestLoader()
    p = plugin.plugins()['plugin']
    self.assertEqual('foo', p.load_plugin_tests(loader))