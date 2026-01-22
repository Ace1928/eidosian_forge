from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional.identity.v2 import common
def test_user_list(self):
    raw_output = self.openstack('user list')
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, common.BASIC_LIST_HEADERS)