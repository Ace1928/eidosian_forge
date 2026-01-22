import sys
import types
import pytest
import pandas.util._test_decorators as td
import pandas
@pytest.fixture
def dummy_backend():
    db = types.ModuleType('pandas_dummy_backend')
    setattr(db, 'plot', lambda *args, **kwargs: 'used_dummy')
    return db