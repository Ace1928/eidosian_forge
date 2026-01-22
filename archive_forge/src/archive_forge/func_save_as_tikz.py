from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def save_as_tikz(self, file_name, colormode='color', width=282.0):
    colors = [pl[-1] for pl in self.polylines]
    tikz = TikZPicture(self.canvas, colors, width)
    for curve in self.curves:
        curve.tikz_draw(tikz, tikz.transform)
    tikz.save(file_name)