from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional.identity.v3 import common
def test_domain_list(self):
    self._create_dummy_domain()
    raw_output = self.openstack('domain list')
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, common.BASIC_LIST_HEADERS)