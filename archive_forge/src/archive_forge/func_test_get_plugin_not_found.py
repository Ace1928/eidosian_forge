from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_get_plugin_not_found(self):
    self.assertIsNone(self.plugin_directory.get_plugin('foo'))