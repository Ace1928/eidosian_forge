import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def test_load_all_skip_tests(self):
    mgr = plugin_manager.PluginManager('heat.tests')
    pm = plugin_manager.PluginMapping('current_test')
    all_items = pm.load_all(mgr)
    for item in current_test_mapping().items():
        self.assertNotIn(item, all_items)