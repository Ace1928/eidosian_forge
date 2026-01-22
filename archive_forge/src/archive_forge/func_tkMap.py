import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkMap(self, *dummy):
    """Cause the opengl widget to redraw itself."""
    self.tkExpose()