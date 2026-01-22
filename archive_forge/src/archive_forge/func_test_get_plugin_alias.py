from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_get_plugin_alias(self):
    directory.add_plugin('foo', fake_plugin)
    self.assertIsNotNone(directory.get_plugin('foo'))