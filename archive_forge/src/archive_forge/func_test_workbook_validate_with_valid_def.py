import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbook_validate_with_valid_def(self):
    wb = self.mistral_admin('workbook-validate', params=self.wb_def)
    wb_valid = self.get_field_value(wb, 'Valid')
    wb_error = self.get_field_value(wb, 'Error')
    self.assertEqual('True', wb_valid)
    self.assertEqual('None', wb_error)