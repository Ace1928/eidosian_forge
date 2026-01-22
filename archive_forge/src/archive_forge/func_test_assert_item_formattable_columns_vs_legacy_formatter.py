import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_assert_item_formattable_columns_vs_legacy_formatter(self):
    expected = [format_columns.DictColumn({'a': 1, 'b': 2}), format_columns.ListColumn(['x', 'y', 'z'])]
    actual = [utils.format_dict({'a': 1, 'b': 2}), utils.format_list(['x', 'y', 'z'])]
    self.assertRaises(AssertionError, self.assertItemEqual, expected, actual)