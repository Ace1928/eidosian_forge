from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_multisection_docstring(self):
    docstring = 'Docstring summary.\n\n    This is the first section of a docstring description.\n\n    This is the second section of a docstring description. This docstring\n    description has just two sections.\n    '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is the first section of a docstring description.\n\nThis is the second section of a docstring description. This docstring\ndescription has just two sections.')
    self.assertEqual(expected_docstring_info, docstring_info)