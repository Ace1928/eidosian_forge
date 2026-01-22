import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_backward_compat_col_showone_no_specify_column(self):
    fake_object = {'id': 'fake-id', 'name': 'fake-name', 'size': 'fake-size'}
    columns = []
    column_map = {'display_name': 'name'}
    results = utils.backward_compat_col_showone(fake_object, columns, column_map)
    self.assertIsInstance(results, dict)
    self.assertNotIn('display_name', results)
    self.assertIn('id', results)
    self.assertIn('name', results)
    self.assertIn('size', results)