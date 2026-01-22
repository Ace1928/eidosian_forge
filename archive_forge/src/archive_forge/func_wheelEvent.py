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
def wheelEvent(self, event):
    if event.pixelDelta().isNull() or QtWidgets.QApplication.instance().platformName() == 'xcb':
        steps = event.angleDelta().y() / 120
    else:
        steps = event.pixelDelta().y()
    if steps:
        MouseEvent('scroll_event', self, *self.mouseEventCoords(event), step=steps, modifiers=self._mpl_modifiers(), guiEvent=event)._process()