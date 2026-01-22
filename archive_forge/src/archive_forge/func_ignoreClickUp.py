from pidWxDc import PiddleWxDc
from wxPython.wx import *
def ignoreClickUp(canvas, x, y):
    canvas.sb.OnClickUp(x, y)