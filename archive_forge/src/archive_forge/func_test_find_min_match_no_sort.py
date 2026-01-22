import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_min_match_no_sort(self):
    items = self._get_test_items()
    sort_str = None
    flair = {}
    expect_items = items
    self.assertEqual(expect_items, list(utils.find_min_match(items, sort_str, **flair)))