from reportlab.graphics.shapes import *
from reportlab.lib import colors
def getDrawing1():
    """Hello World, on a rectangular background"""
    D = Drawing(400, 200)
    D.add(Rect(50, 50, 300, 100, fillColor=colors.yellow))
    D.add(String(180, 100, 'Hello World', fillColor=colors.red))
    return D