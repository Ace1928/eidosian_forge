from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_get_plugin_core(self):
    directory.add_plugin('CORE', fake_plugin)
    self.assertIsNotNone(directory.get_plugin())