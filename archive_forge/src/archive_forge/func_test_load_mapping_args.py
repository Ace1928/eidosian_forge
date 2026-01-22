import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def test_load_mapping_args(self):
    pm = plugin_manager.PluginMapping('args_test', 'baz', 'quux')
    expected = {0: 'baz', 1: 'quux'}
    self.assertEqual(expected, pm.load_from_module(self.module()))