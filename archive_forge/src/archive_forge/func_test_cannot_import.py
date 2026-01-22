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
def test_cannot_import(self):
    self.create_plugin_package('works')
    self.create_plugin_package('fails')
    self.overrideEnv('BRZ_DISABLE_PLUGINS', 'fails')
    self.update_module_paths(['.'])
    import breezy.testingplugins.works as works
    try:
        import breezy.testingplugins.fails as fails
    except ImportError:
        pass
    else:
        self.fail('Loaded blocked plugin: ' + repr(fails))
    self.assertPluginModules({'fails': None, 'works': works})