from osc_lib.tests import utils as test_utils
from osc_lib.utils import columns as column_utils
def test_get_column_definitions(self):
    attr_map = (('id', 'ID', column_utils.LIST_BOTH), ('tenant_id', 'Project', column_utils.LIST_LONG_ONLY), ('name', 'Name', column_utils.LIST_BOTH), ('summary', 'Summary', column_utils.LIST_SHORT_ONLY))
    headers, columns = column_utils.get_column_definitions(attr_map, long_listing=False)
    self.assertEqual(['id', 'name', 'summary'], columns)
    self.assertEqual(['ID', 'Name', 'Summary'], headers)