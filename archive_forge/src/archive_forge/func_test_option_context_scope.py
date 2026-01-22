import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_option_context_scope(self):
    original_value = 60
    context_value = 10
    option_name = 'a'
    cf.register_option(option_name, original_value)
    ctx = cf.option_context(option_name, context_value)
    assert cf.get_option(option_name) == original_value
    with ctx:
        assert cf.get_option(option_name) == context_value
    assert cf.get_option(option_name) == original_value