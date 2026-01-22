from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_numpy_colon_in_description(self):
    docstring = '\n     Greets name.\n\n     Arguments\n     ---------\n     name : str\n         name, default : World\n     arg2 : int\n         arg2, default:None\n     arg3 : bool\n     '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='Greets name.', description=None, args=[ArgInfo(name='name', type='str', description='name, default : World'), ArgInfo(name='arg2', type='int', description='arg2, default:None'), ArgInfo(name='arg3', type='bool', description=None)])
    self.assertEqual(expected_docstring_info, docstring_info)