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
def test_get_help_text_no_carriage_return(self):
    """ModuleHelpTopic.get_help_text adds a 
 if needed."""
    mod = FakeModule('one line of help', 'demo')
    topic = plugin.ModuleHelpTopic(mod)
    self.assertEqual('one line of help\n', topic.get_help_text())