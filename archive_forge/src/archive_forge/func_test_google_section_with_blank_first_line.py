from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_google_section_with_blank_first_line(self):
    docstring = 'Inspired by requests HTTPAdapter docstring.\n\n    :param x: Simple param.\n\n    Usage:\n\n      >>> import requests\n    '
    docstring_info = docstrings.parse(docstring)
    self.assertEqual('Inspired by requests HTTPAdapter docstring.', docstring_info.summary)