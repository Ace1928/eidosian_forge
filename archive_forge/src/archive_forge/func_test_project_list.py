from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v2 import common
def test_project_list(self):
    raw_output = self.openstack('project list')
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, common.BASIC_LIST_HEADERS)