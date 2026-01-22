import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkScale(self, event):
    """Scale the scene.  Achieved by moving the eye position.

        Dragging up zooms in, while dragging down zooms out
        """
    scale = 1 - 0.01 * (event.y - self.ymouse)
    if scale < 0.001:
        scale = 0.001
    elif scale > 1000:
        scale = 1000
    self.distance = self.distance * scale
    self.tkRedraw()
    self.tkRecordMouse(event)