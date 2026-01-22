import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
def test_list_column(self):
    data = ['key1', 'key2']
    col = format_columns.ListColumn(data)
    self.assertEqual(data, col.machine_readable())
    self.assertEqual('key1, key2', col.human_readable())