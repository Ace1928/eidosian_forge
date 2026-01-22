from reportlab.platypus.flowables import *
from reportlab.platypus.flowables import _ContainerSpace
from reportlab.lib.units import inch
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.frames import Frame
from reportlab.rl_config import defaultPageSize, verbose
import reportlab.lib.sequencer
from reportlab.pdfgen import canvas
from reportlab.lib.utils import isSeq, encode_label, decode_label, annotateException, strTypes
import sys
import logging
def myFirstPage(canvas, doc):
    from reportlab.lib.colors import red
    PAGE_HEIGHT = canvas._pagesize[1]
    canvas.saveState()
    canvas.setStrokeColor(red)
    canvas.setLineWidth(5)
    canvas.line(66, 72, 66, PAGE_HEIGHT - 72)
    canvas.setFont(_baseFontNameB, 24)
    canvas.drawString(108, PAGE_HEIGHT - 108, 'TABLE OF CONTENTS DEMO')
    canvas.setFont(_baseFontName, 12)
    canvas.drawString(4 * inch, 0.75 * inch, 'First Page')
    canvas.restoreState()