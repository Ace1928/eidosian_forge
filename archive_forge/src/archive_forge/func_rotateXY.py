import math
import xmllib
from rdkit.sping.pid import Font
from sping.PDF import PDFCanvas
def rotateXY(x, y, theta):
    """Rotate (x,y) by theta degrees.  Got transformation         from page 299 in linear algebra book."""
    radians = theta * math.pi / 180.0
    return (math.cos(radians) * x + math.sin(radians) * y, -(math.sin(radians) * x - math.cos(radians) * y))