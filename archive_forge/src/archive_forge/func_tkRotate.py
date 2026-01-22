import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkRotate(self, event):
    """Perform rotation of scene."""
    self.activate()
    glRotateScene(0.5, self.xcenter, self.ycenter, self.zcenter, event.x, event.y, self.xmouse, self.ymouse)
    self.tkRedraw()
    self.tkRecordMouse(event)