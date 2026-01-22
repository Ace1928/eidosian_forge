import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workbook_get(self):
    created = self.workbook_create(self.wb_with_tags_def)
    wb_name = self.get_field_value(created, 'Name')
    fetched = self.mistral_admin('workbook-get', params=wb_name)
    created_wb_name = self.get_field_value(created, 'Name')
    fetched_wb_name = self.get_field_value(fetched, 'Name')
    self.assertEqual(created_wb_name, fetched_wb_name)
    created_wb_tag = self.get_field_value(created, 'Tags')
    fetched_wb_tag = self.get_field_value(fetched, 'Tags')
    self.assertEqual(created_wb_tag, fetched_wb_tag)