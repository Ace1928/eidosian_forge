from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_unique_plugins(self):
    self.plugin_directory._plugins = {'foo1': fake_plugin, 'foo2': fake_plugin}
    self.assertEqual(1, len(self.plugin_directory.unique_plugins))