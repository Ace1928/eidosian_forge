import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def tkPrint(self, file):
    """Turn the current scene into PostScript via the feedback buffer."""
    self.activate()