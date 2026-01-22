import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_dict_of_list_with_separator(self):
    expected = 'a=a1, a2\nb=b1, b2\nc=c1, c2\ne='
    self.assertEqual(expected, utils.format_dict_of_list({'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'c': ['c1', 'c2'], 'd': None, 'e': []}, separator='\n'))
    self.assertEqual(expected, utils.format_dict_of_list({'c': ['c1', 'c2'], 'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'e': []}, separator='\n'))
    self.assertIsNone(utils.format_dict_of_list(None, separator='\n'))