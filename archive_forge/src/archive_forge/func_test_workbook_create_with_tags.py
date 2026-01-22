import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbook_create_with_tags(self):
    wb = self.workbook_create(self.wb_with_tags_def)
    self.assertIn('tag', self.get_field_value(wb, 'Tags'))