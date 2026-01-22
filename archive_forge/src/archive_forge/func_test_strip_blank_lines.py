from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_strip_blank_lines(self):
    lines = ['   ', '  foo  ', '   ']
    expected_output = ['  foo  ']
    self.assertEqual(expected_output, docstrings._strip_blank_lines(lines))