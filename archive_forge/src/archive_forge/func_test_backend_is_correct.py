import sys
import types
import pytest
import pandas.util._test_decorators as td
import pandas
def test_backend_is_correct(monkeypatch, restore_backend, dummy_backend):
    monkeypatch.setitem(sys.modules, 'pandas_dummy_backend', dummy_backend)
    pandas.set_option('plotting.backend', 'pandas_dummy_backend')
    assert pandas.get_option('plotting.backend') == 'pandas_dummy_backend'
    assert pandas.plotting._core._get_plot_backend('pandas_dummy_backend') is dummy_backend