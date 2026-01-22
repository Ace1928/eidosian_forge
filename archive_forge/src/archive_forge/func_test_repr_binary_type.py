import string
import pandas._config.config as cf
from pandas.io.formats import printing
def test_repr_binary_type(self):
    letters = string.ascii_letters
    try:
        raw = bytes(letters, encoding=cf.get_option('display.encoding'))
    except TypeError:
        raw = bytes(letters)
    b = str(raw.decode('utf-8'))
    res = printing.pprint_thing(b, quote_strings=True)
    assert res == repr(b)
    res = printing.pprint_thing(b, quote_strings=False)
    assert res == b