import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
def testOutputGrabber():
    gr = OutputGrabber()
    for i in range(10):
        print('line', i)
    data = gr.getData()
    gr.close()
    print('Data...', data)