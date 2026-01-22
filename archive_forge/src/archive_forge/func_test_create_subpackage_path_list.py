import importlib.machinery
import sys
from unittest import mock
from heat.common import plugin_loader
import heat.engine
from heat.tests import common
def test_create_subpackage_path_list(self):
    path_list = ['/tmp']
    pkg_name = 'heat.engine.test_path_list'
    self.assertNotIn(pkg_name, sys.modules)
    pkg = plugin_loader.create_subpackage('/tmp', 'heat.engine', 'test_path_list')
    self.assertIn(pkg_name, sys.modules)
    self.assertEqual(sys.modules[pkg_name], pkg)
    self.assertEqual(path_list, pkg.__path__)
    self.assertNotIn(pkg.__path__, path_list)
    self.assertEqual(pkg_name, pkg.__name__)