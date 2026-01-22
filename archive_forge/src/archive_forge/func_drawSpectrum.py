import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def drawSpectrum(canvas):
    """Generates a spectrum plot; illustrates colors and useful application."""
    saver = StateSaver(canvas)

    def plot(f, canvas, offset=0):
        for i in range(0, 100):
            x = float(i) / 100
            canvas.drawLine(i * 3 + offset, 250, i * 3 + offset, 250 - 100 * f(x))

    def genColors(n=100):
        out = [None] * n
        for i in range(n):
            x = float(i) / n
            out[i] = Color(redfunc(x), greenfunc(x), bluefunc(x))
        return out
    colors = genColors(300)
    canvas.drawRect(0, 0, 300, 100, edgeColor=black, fillColor=black)
    for i in range(len(colors)):
        canvas.drawLine(i, 20, i, 80, colors[i])
    canvas.defaultLineColor = red
    plot(redfunc, canvas)
    canvas.defaultLineColor = blue
    plot(bluefunc, canvas, 1)
    canvas.defaultLineColor = green
    plot(greenfunc, canvas, 2)
    return canvas