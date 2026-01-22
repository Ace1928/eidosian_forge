from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_get_unique_plugins(self):
    directory.add_plugin('foo1', fake_plugin)
    directory.add_plugin('foo2', fake_plugin)
    self.assertEqual(1, len(directory.get_unique_plugins()))