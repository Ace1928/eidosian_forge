import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import (
def test_keys_illegal_include_keyword_value(setup_path):
    with ensure_clean_store(setup_path) as store:
        with pytest.raises(ValueError, match="`include` should be either 'pandas' or 'native' but is 'illegal'"):
            store.keys(include='illegal')