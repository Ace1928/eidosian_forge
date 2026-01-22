import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_list_separator(self):
    expected = 'a\nb\nc'
    actual_pre_sorted = utils.format_list(['a', 'b', 'c'], separator='\n')
    actual_unsorted = utils.format_list(['c', 'b', 'a'], separator='\n')
    self.assertEqual(expected, actual_pre_sorted)
    self.assertEqual(expected, actual_unsorted)