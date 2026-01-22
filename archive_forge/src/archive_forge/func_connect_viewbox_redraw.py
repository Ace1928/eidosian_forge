import jupyter_rfb
import numpy as np
from .. import functions as fn
from .. import graphicsItems, widgets
from ..Qt import QtCore, QtGui
def connect_viewbox_redraw(vb, request_draw):
    vb.sigRangeChanged.connect(request_draw)
    vb.sigRangeChangedManually.connect(request_draw)
    vb.sigStateChanged.connect(request_draw)
    vb.sigTransformChanged.connect(request_draw)