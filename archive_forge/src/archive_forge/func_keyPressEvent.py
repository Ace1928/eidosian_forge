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
def keyPressEvent(self, event):
    key = self._get_key(event)
    if key is not None:
        KeyEvent('key_press_event', self, key, *self.mouseEventCoords(), guiEvent=event)._process()