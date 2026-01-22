import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_dict(self):
    expected = "a='b', c='d', e='f'"
    self.assertEqual(expected, utils.format_dict({'a': 'b', 'c': 'd', 'e': 'f'}))
    self.assertEqual(expected, utils.format_dict({'e': 'f', 'c': 'd', 'a': 'b'}))
    self.assertIsNone(utils.format_dict(None))