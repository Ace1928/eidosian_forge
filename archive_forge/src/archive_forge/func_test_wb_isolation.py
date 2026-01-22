from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_wb_isolation(self):
    wb = self.workbook_create(self.wb_def)
    wb_name = self.get_field_value(wb, 'Name')
    wbs = self.mistral_admin('workbook-list')
    self.assertIn(wb_name, [w['Name'] for w in wbs])
    alt_wbs = self.mistral_alt_user('workbook-list')
    self.assertNotIn(wb_name, [w['Name'] for w in alt_wbs])