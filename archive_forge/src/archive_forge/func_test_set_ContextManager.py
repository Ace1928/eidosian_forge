import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_set_ContextManager(self):

    def eq(val):
        assert cf.get_option('a') == val
    cf.register_option('a', 0)
    eq(0)
    with cf.option_context('a', 15):
        eq(15)
        with cf.option_context('a', 25):
            eq(25)
        eq(15)
    eq(0)
    cf.set_option('a', 17)
    eq(17)

    @cf.option_context('a', 123)
    def f():
        eq(123)
    f()