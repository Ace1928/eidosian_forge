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
def test_test_suite_gives_test_suite_result(self):
    source = "def test_suite(): return 'foo'"
    self.setup_plugin(source)
    p = plugin.plugins()['plugin']
    self.assertEqual('foo', p.test_suite())