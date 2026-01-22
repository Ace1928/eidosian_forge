import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
def configure_subplots(self):
    if self._subplot_dialog is None:
        self._subplot_dialog = SubplotToolQt(self.canvas.figure, self.canvas.parent())
        self.canvas.mpl_connect('close_event', lambda e: self._subplot_dialog.reject())
    self._subplot_dialog.update_from_current_subplotpars()
    self._subplot_dialog.show()
    return self._subplot_dialog