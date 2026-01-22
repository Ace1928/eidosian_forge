import copy
import importlib
import os
import signal
import sys
from datetime import date, datetime
from unittest import mock
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_fig_sigint_override(qt_core):
    from matplotlib.backends.backend_qt5 import _BackendQT5
    plt.figure()
    event_loop_handler = None

    def fire_signal_and_quit():
        nonlocal event_loop_handler
        event_loop_handler = signal.getsignal(signal.SIGINT)
        qt_core.QCoreApplication.exit()
    qt_core.QTimer.singleShot(0, fire_signal_and_quit)
    original_handler = signal.getsignal(signal.SIGINT)

    def custom_handler(signum, frame):
        pass
    signal.signal(signal.SIGINT, custom_handler)
    try:
        matplotlib.backends.backend_qt._BackendQT.mainloop()
        assert event_loop_handler != custom_handler
        assert signal.getsignal(signal.SIGINT) == custom_handler
        for custom_handler in (signal.SIG_DFL, signal.SIG_IGN):
            qt_core.QTimer.singleShot(0, fire_signal_and_quit)
            signal.signal(signal.SIGINT, custom_handler)
            _BackendQT5.mainloop()
            assert event_loop_handler == custom_handler
            assert signal.getsignal(signal.SIGINT) == custom_handler
    finally:
        signal.signal(signal.SIGINT, original_handler)