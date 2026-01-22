import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
@_isolated_tk_test(success_count=1)
def test_canvas_focus():
    import tkinter as tk
    import matplotlib.pyplot as plt
    success = []

    def check_focus():
        tkcanvas = fig.canvas.get_tk_widget()
        if not tkcanvas.winfo_viewable():
            tkcanvas.wait_visibility()
        if tkcanvas.focus_lastfor() == tkcanvas:
            success.append(True)
        plt.close()
        root.destroy()
    root = tk.Tk()
    fig = plt.figure()
    plt.plot([1, 2, 3])
    root.after(0, plt.show)
    root.after(100, check_focus)
    root.mainloop()
    if success:
        print('success')