import math
import xmllib
from rdkit.sping.pid import Font
from sping.PDF import PDFCanvas
def calcNewY(self, font, y):
    """Returns a new y coordinate depending on its                 whether the string is a sub and super script."""
    if self.sub == 1:
        return y + font.size * subFraction
    elif self.super == 1:
        return y - font.size * superFraction
    else:
        return y