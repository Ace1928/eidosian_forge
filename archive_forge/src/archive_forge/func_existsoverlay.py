import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def existsoverlay(self):
    return self.tk.getboolean(self.tk.call(self._w, 'existsoverlay'))