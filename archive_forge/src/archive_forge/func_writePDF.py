import os, sys
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import asNative, base64_decodebytes
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import *
import unittest
from reportlab.rl_config import register_reset
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def writePDF(drawings):
    """Create and save a PDF file containing some drawings."""
    pdfPath = os.path.splitext(sys.argv[0])[0] + '.pdf'
    c = Canvas(pdfPath)
    c.setFont(_FONTS[0], 32)
    c.drawString(80, 750, 'ReportLab Graphics-Shapes Test')
    c.setFont(_FONTS[0], 12)
    y = 740
    i = 1
    for drawing, docstring, funcname in drawings:
        if y < 300:
            c.showPage()
            y = 740
        y = y - 30
        c.setFont(_FONTS[2], 12)
        c.drawString(80, y, '%s (#%d)' % (funcname, i))
        c.setFont(_FONTS[0], 12)
        y = y - 14
        textObj = c.beginText(80, y)
        textObj.textLines(docstring)
        c.drawText(textObj)
        y = textObj.getY()
        y = y - drawing.height
        drawing.drawOn(c, 80, y)
        i = i + 1
    c.save()
    print('wrote %s ' % pdfPath)