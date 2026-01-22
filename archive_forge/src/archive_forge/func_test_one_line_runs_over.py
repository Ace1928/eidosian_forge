from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_one_line_runs_over(self):
    docstring = 'A one line docstring that is both a little too verbose and a little too long\n    so it runs onto a second line.\n    '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='A one line docstring that is both a little too verbose and a little too long so it runs onto a second line.')
    self.assertEqual(expected_docstring_info, docstring_info)