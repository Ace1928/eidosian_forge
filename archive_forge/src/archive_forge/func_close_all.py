from __future__ import division
import numpy as np
from pygsp import utils
def close_all():
    """
    Close all opened windows.

    """
    global _qtg_windows
    for window in _qtg_windows:
        window.close()
    _qtg_windows = []
    global _qtg_widgets
    for widget in _qtg_widgets:
        widget.close()
    _qtg_widgets = []
    global _plt_figures
    for fig in _plt_figures:
        plt = _import_plt()
        plt.close(fig)
    _plt_figures = []