import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkExpose(self, *dummy):
    """Redraw the widget.
        Make it active, update tk events, call redraw procedure and
        swap the buffers.  Note: swapbuffers is clever enough to
        only swap double buffered visuals."""
    self.activate()
    if not self.initialised:
        self.basic_lighting()
        self.initialised = 1
    self.tkRedraw()