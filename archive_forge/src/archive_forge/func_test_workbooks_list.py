import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbooks_list(self):
    workbooks = self.parser.listing(self.mistral('workbook-list'))
    self.assertTableStruct(workbooks, ['Name', 'Tags', 'Created at', 'Updated at'])