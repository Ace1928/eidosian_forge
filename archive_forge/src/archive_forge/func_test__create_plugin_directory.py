from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test__create_plugin_directory(self):
    self.assertIsNotNone(directory._create_plugin_directory())