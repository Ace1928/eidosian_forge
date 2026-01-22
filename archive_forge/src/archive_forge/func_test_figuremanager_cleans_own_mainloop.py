import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
@pytest.mark.skipif(platform.python_implementation() != 'CPython', reason='PyPy does not support Tkinter threading: https://foss.heptapod.net/pypy/pypy/-/issues/1929')
@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=1)
def test_figuremanager_cleans_own_mainloop():
    import tkinter
    import time
    import matplotlib.pyplot as plt
    import threading
    from matplotlib.cbook import _get_running_interactive_framework
    root = tkinter.Tk()
    plt.plot([1, 2, 3], [1, 2, 5])

    def target():
        while not 'tk' == _get_running_interactive_framework():
            time.sleep(0.01)
        plt.close()
        if show_finished_event.wait():
            print('success')
    show_finished_event = threading.Event()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    plt.show(block=True)
    show_finished_event.set()
    thread.join()