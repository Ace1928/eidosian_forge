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
def test_contruct(self):
    """Construction takes the module to document."""
    mod = FakeModule('foo', 'foo')
    topic = plugin.ModuleHelpTopic(mod)
    self.assertEqual(mod, topic.module)