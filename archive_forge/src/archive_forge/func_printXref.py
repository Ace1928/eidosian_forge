import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
def printXref(self):
    self.startxref = sys.stdout.tell()
    print('xref')
    print(0, len(self.objects) + 1)
    print('0000000000 65535 f')
    for pos in self.xref:
        print('%0.10d 00000 n' % pos)