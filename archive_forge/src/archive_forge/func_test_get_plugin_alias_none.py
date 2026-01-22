from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_get_plugin_alias_none(self):
    self.assertIsNone(directory.get_plugin('foo'))