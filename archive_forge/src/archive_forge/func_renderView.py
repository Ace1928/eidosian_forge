from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
import atexit
import enum
import mmap
import os
import sys
import tempfile
from .. import Qt
from .. import CONFIG_OPTIONS
from .. import multiprocess as mp
from .GraphicsView import GraphicsView
def renderView(self):
    if self.img is None:
        if self.width() == 0 or self.height() == 0:
            return
        dpr = self.devicePixelRatioF()
        iwidth = int(self.width() * dpr)
        iheight = int(self.height() * dpr)
        size = iwidth * iheight * 4
        if size > self.shm.size():
            try:
                self.shm.resize(size)
            except SystemError:
                self.shm.close()
                fd = self.shmFile.fileno()
                os.ftruncate(fd, size)
                self.shm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        if QT_LIB.startswith('PyQt'):
            img_ptr = int(Qt.sip.voidptr(self.shm))
        else:
            img_ptr = self.shm
        self.img = QtGui.QImage(img_ptr, iwidth, iheight, QtGui.QImage.Format.Format_RGB32)
        self.img.setDevicePixelRatio(dpr)
        self.img.fill(4294967295)
        p = QtGui.QPainter(self.img)
        self.render(p, self.viewRect(), self.rect())
        p.end()
        self.sceneRendered.emit((iwidth, iheight, self.shm.size()))