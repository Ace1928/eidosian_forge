import sys
import types
import pytest
import pandas.util._test_decorators as td
import pandas
def test_backend_is_not_module():
    msg = "Could not find plotting backend 'not_an_existing_module'."
    with pytest.raises(ValueError, match=msg):
        pandas.set_option('plotting.backend', 'not_an_existing_module')
    assert pandas.options.plotting.backend == 'matplotlib'