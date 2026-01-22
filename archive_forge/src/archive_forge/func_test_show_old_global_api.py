import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends import backend_template
from matplotlib.backends.backend_template import (
def test_show_old_global_api(monkeypatch):
    mpl_test_backend = SimpleNamespace(**vars(backend_template))
    mock_show = MagicMock()
    monkeypatch.setattr(mpl_test_backend, 'show', mock_show, raising=False)
    monkeypatch.setitem(sys.modules, 'mpl_test_backend', mpl_test_backend)
    mpl.use('module://mpl_test_backend')
    plt.show()
    mock_show.assert_called_with()