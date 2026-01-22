import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
def drawAt(self, pos, ev=None):
    if self.axisOrder == 'col-major':
        pos = [int(pos.x()), int(pos.y())]
    else:
        pos = [int(pos.y()), int(pos.x())]
    dk = self.drawKernel
    kc = self.drawKernelCenter
    sx = [0, dk.shape[0]]
    sy = [0, dk.shape[1]]
    tx = [pos[0] - kc[0], pos[0] - kc[0] + dk.shape[0]]
    ty = [pos[1] - kc[1], pos[1] - kc[1] + dk.shape[1]]
    for i in [0, 1]:
        dx1 = -min(0, tx[i])
        dx2 = min(0, self.image.shape[0] - tx[i])
        tx[i] += dx1 + dx2
        sx[i] += dx1 + dx2
        dy1 = -min(0, ty[i])
        dy2 = min(0, self.image.shape[1] - ty[i])
        ty[i] += dy1 + dy2
        sy[i] += dy1 + dy2
    ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
    ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
    mask = self.drawMask
    src = dk
    if isinstance(self.drawMode, Callable):
        self.drawMode(dk, self.image, mask, ss, ts, ev)
    else:
        src = src[ss]
        if self.drawMode == 'set':
            if mask is not None:
                mask = mask[ss]
                self.image[ts] = self.image[ts] * (1 - mask) + src * mask
            else:
                self.image[ts] = src
        elif self.drawMode == 'add':
            self.image[ts] += src
        else:
            raise Exception("Unknown draw mode '%s'" % self.drawMode)
        self.updateImage()