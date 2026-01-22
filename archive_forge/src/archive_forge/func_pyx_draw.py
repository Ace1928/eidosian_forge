from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def pyx_draw(self, canvas, transform):
    XY = [transform(xy) for xy in self.bezier()]
    arc_parts = [pyx.path.moveto(*XY[0])]
    for i in range(1, len(XY), 3):
        arc_parts.append(pyx.path.curveto(XY[i][0], XY[i][1], XY[i + 1][0], XY[i + 1][1], XY[i + 2][0], XY[i + 2][1]))
        style = [pyx.style.linewidth(4), pyx.style.linecap.round, pyx.color.rgbfromhexstring(self.color)]
        path = pyx.path.path(*arc_parts)
        canvas.stroke(path, style)