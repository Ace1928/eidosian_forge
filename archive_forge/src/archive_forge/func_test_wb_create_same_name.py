import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_wb_create_same_name(self):
    self.workbook_create(self.wb_def)
    self.assertRaises(exceptions.CommandFailed, self.workbook_create, self.wb_def)