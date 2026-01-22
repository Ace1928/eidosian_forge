from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbook_name_uniqueness(self):
    self.workbook_create(self.wb_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-create', params='{0}'.format(self.wb_def))
    self.workbook_create(self.wb_def, admin=False)
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workbook-create', params='{0}'.format(self.wb_def))