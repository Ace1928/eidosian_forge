import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_backward_compat_col_lister(self):
    fake_col_headers = ['ID', 'Name', 'Size']
    columns = ['Display Name']
    column_map = {'Display Name': 'Name'}
    results = utils.backward_compat_col_lister(fake_col_headers, columns, column_map)
    self.assertIsInstance(results, list)
    self.assertIn('Display Name', results)
    self.assertNotIn('Name', results)
    self.assertIn('ID', results)
    self.assertIn('Size', results)