import sys
import math, random
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *
def randomColor(self):
    red = random.randrange(205, 256)
    green = random.randrange(205, 256)
    blue = random.randrange(205, 256)
    alpha = random.randrange(91, 192)
    return QColor(red, green, blue, alpha)