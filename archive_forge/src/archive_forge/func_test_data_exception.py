from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_data_exception(self):
    os = StringIO()
    data = {'a': 0, 'b': 1, 'c': 3}

    def _data_gen(i, j):
        if i == 'b':
            raise ValueError('invalid')
        return (j, i * (j + 1))
    tabular_writer(os, '', data.items(), ['i', 'j'], _data_gen)
    ref = u'\nKey : i    : j\n  a :    0 :    a\n  b : None : None\n  c :    3 : cccc\n'
    self.assertEqual(ref.strip(), os.getvalue().strip())