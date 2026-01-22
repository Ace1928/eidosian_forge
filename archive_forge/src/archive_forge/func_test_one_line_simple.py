from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_one_line_simple(self):
    docstring = 'A simple one line docstring.'
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='A simple one line docstring.')
    self.assertEqual(expected_docstring_info, docstring_info)