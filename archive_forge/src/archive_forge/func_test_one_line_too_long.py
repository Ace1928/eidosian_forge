from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_one_line_too_long(self):
    docstring = 'A one line docstring that is both a little too verbose and a little too long so it keeps going well beyond a reasonable length for a one-liner.\n    '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='A one line docstring that is both a little too verbose and a little too long so it keeps going well beyond a reasonable length for a one-liner.')
    self.assertEqual(expected_docstring_info, docstring_info)