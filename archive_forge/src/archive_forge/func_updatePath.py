from .. import functions as fn
from ..Qt import QtGui, QtWidgets
from .PlotCurveItem import PlotCurveItem
from .PlotDataItem import PlotDataItem
def updatePath(self):
    if self.curves is None:
        self.setPath(QtGui.QPainterPath())
        return
    paths = []
    for c in self.curves:
        if isinstance(c, PlotDataItem):
            paths.append(c.curve.getPath())
        elif isinstance(c, PlotCurveItem):
            paths.append(c.getPath())
    path = QtGui.QPainterPath()
    transform = QtGui.QTransform()
    ps1 = paths[0].toSubpathPolygons(transform)
    ps2 = paths[1].toReversed().toSubpathPolygons(transform)
    ps2.reverse()
    if len(ps1) == 0 or len(ps2) == 0:
        self.setPath(QtGui.QPainterPath())
        return
    for p1, p2 in zip(ps1, ps2):
        path.addPolygon(p1 + p2)
    self.setPath(path)