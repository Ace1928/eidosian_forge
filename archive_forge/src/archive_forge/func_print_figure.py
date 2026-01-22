import ctypes
from matplotlib.transforms import Bbox
from .qt_compat import QT_API, QtCore, QtGui
from .backend_agg import FigureCanvasAgg
from .backend_qt import _BackendQT, FigureCanvasQT
from .backend_qt import (  # noqa: F401 # pylint: disable=W0611
def print_figure(self, *args, **kwargs):
    super().print_figure(*args, **kwargs)
    self._draw_pending = True