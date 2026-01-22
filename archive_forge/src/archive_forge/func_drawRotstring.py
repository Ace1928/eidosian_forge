import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def drawRotstring(canvas):
    """Draws rotated strings."""
    saver = StateSaver(canvas)
    canvas.defaultFont = Font(bold=1)
    canvas.defaultLineColor = (blue + white) / 2
    canvas.drawLine(0, 150, 300, 150)
    canvas.drawLine(150, 0, 150, 300)
    s = ' __albatros at '
    w = canvas.stringWidth(s)
    canvas.drawEllipse(150 - w, 150 - w, 150 + w, 150 + w, fillColor=transparent)
    colors = [red, orange, yellow, green, blue, purple]
    cnum = 0
    for ang in range(0, 359, 30):
        canvas.defaultLineColor = colors[cnum]
        s2 = s + str(ang)
        canvas.drawString(s2, 150, 150, angle=ang)
        cnum = (cnum + 1) % len(colors)
    canvas.drawString('This is  a\nrotated\nmulti-line string!!!', 350, 100, angle=-90, font=Font(underline=1))
    return canvas