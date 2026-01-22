import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def setcolor(self, index, red, green, blue):
    self.tk.call(self._w, 'setcolor', index, red, green, blue)