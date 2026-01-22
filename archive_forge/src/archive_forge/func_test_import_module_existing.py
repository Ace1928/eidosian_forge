import importlib.machinery
import sys
from unittest import mock
from heat.common import plugin_loader
import heat.engine
from heat.tests import common
def test_import_module_existing(self):
    import heat.engine.service
    existing = heat.engine.service
    importer = importlib.machinery.FileFinder(heat.engine.__path__[0])
    loaded = plugin_loader._import_module(importer, 'heat.engine.service', heat.engine)
    self.assertIs(existing, loaded)