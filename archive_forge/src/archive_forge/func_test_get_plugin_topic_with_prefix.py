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
def test_get_plugin_topic_with_prefix(self):
    """Searching for plugins/demo_module returns help."""
    index = plugin.PluginsHelpIndex()
    self.assertFalse('breezy.plugins.demo_module' in sys.modules)
    demo_module = FakeModule('', 'breezy.plugins.demo_module')
    sys.modules['breezy.plugins.demo_module'] = demo_module
    try:
        topics = index.get_topics('plugins/demo_module')
        self.assertEqual(1, len(topics))
        self.assertIsInstance(topics[0], plugin.ModuleHelpTopic)
        self.assertEqual(demo_module, topics[0].module)
    finally:
        del sys.modules['breezy.plugins.demo_module']