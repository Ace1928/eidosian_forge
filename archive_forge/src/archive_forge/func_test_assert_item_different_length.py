import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_assert_item_different_length(self):
    expected = ['a', 'b', 'c']
    actual = ['a', 'b']
    self.assertRaises(AssertionError, self.assertItemEqual, expected, actual)