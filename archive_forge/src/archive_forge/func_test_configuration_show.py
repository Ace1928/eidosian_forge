import os
from openstackclient.common import configuration
from openstackclient.tests.functional import base
def test_configuration_show(self):
    raw_output = self.openstack('configuration show', cloud=None)
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, BASIC_CONFIG_HEADERS)
    cmd_output = self.openstack('configuration show', cloud=None, parse_output=True)
    self.assertNotIn('auth.password', cmd_output)