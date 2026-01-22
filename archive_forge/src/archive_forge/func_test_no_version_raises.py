import sys
import types
import pytest
from pandas.compat._optional import (
import pandas._testing as tm
def test_no_version_raises(monkeypatch):
    name = 'fakemodule'
    module = types.ModuleType(name)
    sys.modules[name] = module
    monkeypatch.setitem(VERSIONS, name, '1.0.0')
    with pytest.raises(ImportError, match="Can't determine .* fakemodule"):
        import_optional_dependency(name)