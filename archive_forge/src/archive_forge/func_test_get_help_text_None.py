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
def test_get_help_text_None(self):
    """A ModuleHelpTopic returns the docstring for get_help_text."""
    mod = FakeModule(None, 'demo')
    topic = plugin.ModuleHelpTopic(mod)
    self.assertEqual("Plugin 'demo' has no docstring.\n", topic.get_help_text())