import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
def glTranslateScene(s, x, y, mousex, mousey):
    glMatrixMode(GL_MODELVIEW)
    mat = glGetDoublev(GL_MODELVIEW_MATRIX)
    glLoadIdentity()
    glTranslatef(s * (x - mousex), s * (mousey - y), 0.0)
    glMultMatrixd(mat)