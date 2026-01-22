import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def set_background(self, r, g, b):
    """Change the background colour of the widget."""
    self.r_back = r
    self.g_back = g
    self.b_back = b
    self.tkRedraw()