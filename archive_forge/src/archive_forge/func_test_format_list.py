import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_list(self):
    expected = 'a, b, c'
    self.assertEqual(expected, utils.format_list(['a', 'b', 'c']))
    self.assertEqual(expected, utils.format_list(['c', 'b', 'a']))
    self.assertIsNone(utils.format_list(None))