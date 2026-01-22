import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
def generatePicture(self):
    self.picture = QtGui.QPicture()
    p = QtGui.QPainter(self.picture)
    p.setPen(pg.mkPen('w'))
    w = (self.data[1][0] - self.data[0][0]) / 3.0
    for t, open, close, min, max in self.data:
        p.drawLine(QtCore.QPointF(t, min), QtCore.QPointF(t, max))
        if open > close:
            p.setBrush(pg.mkBrush('r'))
        else:
            p.setBrush(pg.mkBrush('g'))
        p.drawRect(QtCore.QRectF(t - w, open, w * 2, close - open))
    p.end()