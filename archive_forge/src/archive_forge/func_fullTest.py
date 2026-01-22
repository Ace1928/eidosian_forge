import sys, time
from reportlab import Version as __RL_Version__
from reportlab.graphics.barcode.common import *
from reportlab.graphics.barcode.code39 import *
from reportlab.graphics.barcode.code93 import *
from reportlab.graphics.barcode.code128 import *
from reportlab.graphics.barcode.usps import *
from reportlab.graphics.barcode.usps4s import USPS_4State
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.barcode.dmtx import DataMatrixWidget, pylibdmtx
from reportlab.platypus import Spacer, SimpleDocTemplate, PageBreak
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.flowables import XBox, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics.barcode import getCodeNames, createBarcodeDrawing, createBarcodeImageInMemory
def fullTest(fileName='test_full.pdf'):
    """Creates large-ish test document with a variety of parameters"""
    story = []
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleH = styles['Heading1']
    styleH2 = styles['Heading2']
    story = []
    story.append(Paragraph('ReportLab %s Barcode Test Suite - full output' % __RL_Version__, styleH))
    story.append(Paragraph('Generated at %s' % time.ctime(time.time()), styleN))
    story.append(Paragraph('About this document', styleH2))
    story.append(Paragraph('History and Status', styleH2))
    story.append(Paragraph('\n        This is the test suite and docoumentation for the ReportLab open source barcode API.\n        ', styleN))
    story.append(Paragraph('\n        Several years ago Ty Sarna contributed a barcode module to the ReportLab community.\n        Several of the codes were used by him in hiw work and to the best of our knowledge\n        this was correct.  These were written as flowable objects and were available in PDFs,\n        but not in our graphics framework.  However, we had no knowledge of barcodes ourselves\n        and did not advertise or extend the package.\n        ', styleN))
    story.append(Paragraph('\n        We "wrapped" the barcodes to be usable within our graphics framework; they are now available\n        as Drawing objects which can be rendered to EPS files or bitmaps.  For the last 2 years this\n        has been available in our Diagra and Report Markup Language products.  However, we did not\n        charge separately and use was on an "as is" basis.\n        ', styleN))
    story.append(Paragraph('\n        A major licensee of our technology has kindly agreed to part-fund proper productisation\n        of this code on an open source basis in Q1 2006.  This has involved addition of EAN codes\n        as well as a proper testing program.  Henceforth we intend to publicise the code more widely,\n        gather feedback, accept contributions of code and treat it as "supported".  \n        ', styleN))
    story.append(Paragraph('\n        This involved making available both downloads and testing resources.  This PDF document\n        is the output of the current test suite.  It contains codes you can scan (if you use a nice sharp\n        laser printer!), and will be extended over coming weeks to include usage examples and notes on\n        each barcode and how widely tested they are.  This is being done through documentation strings in\n        the barcode objects themselves so should always be up to date.\n        ', styleN))
    story.append(Paragraph('Usage examples', styleH2))
    story.append(Paragraph('\n        To be completed\n        ', styleN))
    story.append(Paragraph('The codes', styleH2))
    story.append(Paragraph('\n        Below we show a scannable code from each barcode, with and without human-readable text.\n        These are magnified about 2x from the natural size done by the original author to aid\n        inspection.  This will be expanded to include several test cases per code, and to add\n        explanations of checksums.  Be aware that (a) if you enter numeric codes which are too\n        short they may be prefixed for you (e.g. "123" for an 8-digit code becomes "00000123"),\n        and that the scanned results and readable text will generally include extra checksums\n        at the end.\n        ', styleN))
    codeNames = getCodeNames()
    from reportlab.lib.utils import flatten
    width = [float(x[8:]) for x in sys.argv if x.startswith('--width=')]
    height = [float(x[9:]) for x in sys.argv if x.startswith('--height=')]
    isoScale = [int(x[11:]) for x in sys.argv if x.startswith('--isoscale=')]
    options = {}
    if width:
        options['width'] = width[0]
    if height:
        options['height'] = height[0]
    if isoScale:
        options['isoScale'] = isoScale[0]
    scales = [x[8:].split(',') for x in sys.argv if x.startswith('--scale=')]
    scales = list(map(float, scales and flatten(scales) or [1]))
    scales = list(map(float, scales and flatten(scales) or [1]))
    for scale in scales:
        story.append(PageBreak())
        story.append(Paragraph('Scale = %.1f' % scale, styleH2))
        story.append(Spacer(36, 12))
        for codeName in codeNames:
            s = [Paragraph('Code: ' + codeName, styleH2)]
            for hr in (0, 1):
                s.append(Spacer(36, 12))
                dr = createBarcodeDrawing(codeName, humanReadable=hr, **options)
                dr.renderScale = scale
                s.append(dr)
                s.append(Spacer(36, 12))
            s.append(Paragraph('Barcode should say: ' + dr._bc.value, styleN))
            story.append(KeepTogether(s))
    SimpleDocTemplate(fileName).build(story)
    print('created', fileName)