import pytest
import importlib
def test_compat_warn():
    with pytest.warns(DeprecationWarning):
        import toolz.compatibility
        importlib.reload(toolz.compatibility)