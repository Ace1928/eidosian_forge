import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch('cliff.utils.terminal_width')
def test_max_width_10(self, tw):
    width = tw.return_value = 10
    actual = _table_tester_helper(self._col_names, self._col_data, extra_args=['--fit-width'])
    self.assertEqual(self._expected_mv[width], actual)
    expected_width = 11 * 3 + 1
    self.assertEqual(expected_width, len(actual.splitlines()[0]))