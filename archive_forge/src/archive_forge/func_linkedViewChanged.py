import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def linkedViewChanged(self, view, axis):
    if self.linksBlocked or view is None:
        return
    vr = view.viewRect()
    vg = view.screenGeometry()
    sg = self.screenGeometry()
    if vg is None or sg is None:
        return
    view.blockLink(True)
    try:
        if axis == ViewBox.XAxis:
            overlap = min(sg.right(), vg.right()) - max(sg.left(), vg.left())
            if overlap < min(vg.width() / 3, sg.width() / 3):
                x1 = vr.left()
                x2 = vr.right()
            else:
                upp = float(vr.width()) / vg.width()
                if self.xInverted():
                    x1 = vr.left() + (sg.right() - vg.right()) * upp
                else:
                    x1 = vr.left() + (sg.x() - vg.x()) * upp
                x2 = x1 + sg.width() * upp
            self.enableAutoRange(ViewBox.XAxis, False)
            self.setXRange(x1, x2, padding=0)
        else:
            overlap = min(sg.bottom(), vg.bottom()) - max(sg.top(), vg.top())
            if overlap < min(vg.height() / 3, sg.height() / 3):
                y1 = vr.top()
                y2 = vr.bottom()
            else:
                upp = float(vr.height()) / vg.height()
                if self.yInverted():
                    y2 = vr.bottom() + (sg.bottom() - vg.bottom()) * upp
                else:
                    y2 = vr.bottom() + (sg.top() - vg.top()) * upp
                y1 = y2 - sg.height() * upp
            self.enableAutoRange(ViewBox.YAxis, False)
            self.setYRange(y1, y2, padding=0)
    finally:
        view.blockLink(False)