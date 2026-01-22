from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_project_list_with_domain(self):
    project_name = self._create_dummy_project()
    raw_output = self.openstack('project list --domain %s' % self.domain_name)
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, common.BASIC_LIST_HEADERS)
    self.assertIn(project_name, raw_output)
    self.assertGreater(len(items), 0)