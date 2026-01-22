import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch('cliff.utils.terminal_width')
def test_70(self, tw):
    tw.return_value = 70
    c = ('field_name', 'a_really_long_field_name')
    d = ('the value', 'a value significantly longer than the field')
    expected = textwrap.dedent('        +--------------------------+-----------------------------------------+\n        | Field                    | Value                                   |\n        +--------------------------+-----------------------------------------+\n        | field_name               | the value                               |\n        | a_really_long_field_name | a value significantly longer than the   |\n        |                          | field                                   |\n        +--------------------------+-----------------------------------------+\n        ')
    self.assertEqual(expected, _table_tester_helper(c, d, extra_args=['--fit-width']))