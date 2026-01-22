import string
import pandas._config.config as cf
from pandas.io.formats import printing
def test_adjoin(self):
    data = [['a', 'b', 'c'], ['dd', 'ee', 'ff'], ['ggg', 'hhh', 'iii']]
    expected = 'a  dd  ggg\nb  ee  hhh\nc  ff  iii'
    adjoined = printing.adjoin(2, *data)
    assert adjoined == expected