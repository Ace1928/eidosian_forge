import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def showoverlay(self):
    self.tk.call(self._w, 'showoverlay')