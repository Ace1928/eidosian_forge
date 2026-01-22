import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_reset_option_all(self):
    cf.register_option('a', 1, 'doc', validator=cf.is_int)
    cf.register_option('b.c', 'hullo', 'doc2', validator=cf.is_str)
    assert cf.get_option('a') == 1
    assert cf.get_option('b.c') == 'hullo'
    cf.set_option('a', 2)
    cf.set_option('b.c', 'wurld')
    assert cf.get_option('a') == 2
    assert cf.get_option('b.c') == 'wurld'
    cf.reset_option('all')
    assert cf.get_option('a') == 1
    assert cf.get_option('b.c') == 'hullo'