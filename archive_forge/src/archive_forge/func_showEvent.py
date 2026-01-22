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
def showEvent(self, event):
    window = self.window().windowHandle()
    window.screenChanged.connect(self._update_screen)
    self._update_screen(window.screen())