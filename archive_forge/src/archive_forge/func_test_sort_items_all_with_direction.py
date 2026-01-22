import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_sort_items_all_with_direction(self):
    items = self._get_test_items()
    sort_str = 'a:desc,b:desc'
    expect_items = [items[2], items[3], items[1], items[0]]
    self.assertEqual(expect_items, utils.sort_items(items, sort_str))