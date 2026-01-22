from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def save_as_pdf(self, file_name, colormode='color', width=312.0):
    """
        Save the smooth link diagram as a PDF file.
        Accepts options colormode and width.
        The colormode (currently ignored) must be 'color', 'gray', or 'mono'; default is 'color'.
        The width option sets the width of the figure in points.
        The default width is 312pt = 4.33in = 11cm .
        """
    PDF = PDFPicture(self.canvas, width)
    for curve in self.curves:
        curve.pyx_draw(PDF.canvas, PDF.transform)
    PDF.save(file_name)