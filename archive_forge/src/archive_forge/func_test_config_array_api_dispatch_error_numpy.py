import builtins
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed
def test_config_array_api_dispatch_error_numpy(monkeypatch):
    """Check error when NumPy is too old"""
    orig_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == 'array_api_compat':
            return object()
        return orig_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', mocked_import)
    monkeypatch.setattr(sklearn.utils._array_api.numpy, '__version__', '1.20')
    with pytest.raises(ImportError, match='NumPy must be 1.21 or newer'):
        with config_context(array_api_dispatch=True):
            pass
    with pytest.raises(ImportError, match='NumPy must be 1.21 or newer'):
        set_config(array_api_dispatch=True)