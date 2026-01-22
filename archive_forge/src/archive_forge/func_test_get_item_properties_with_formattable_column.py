import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_get_item_properties_with_formattable_column(self):
    formatters = {'attr': format_columns.ListColumn}
    res_attr = self._test_get_item_properties_with_formatter(formatters)
    self.assertIsInstance(res_attr, format_columns.ListColumn)