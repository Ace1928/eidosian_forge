import math
import xmllib
from rdkit.sping.pid import Font
from sping.PDF import PDFCanvas
def parseSegments(self, s):
    """Given a formatted string will return a list of                 StringSegment objects with their calculated widths."""
    self.feed('<formattedstring>' + s + '</formattedstring>')
    self.close()
    self.reset()
    segmentlist = self.segmentlist
    self.segmentlist = []
    return segmentlist