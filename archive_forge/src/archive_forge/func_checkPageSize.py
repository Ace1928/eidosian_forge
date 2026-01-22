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
def checkPageSize(self, canv, doc):
    """This gets called by the template framework
        If canv size != template size then the canv size is set to
        the template size or if that's not available to the
        doc size.
        """
    cp = None
    dp = None
    sp = None
    if canv._pagesize:
        cp = list(map(int, canv._pagesize))
    if self.pagesize:
        sp = list(map(int, self.pagesize))
    if doc.pagesize:
        dp = list(map(int, doc.pagesize))
    if cp != sp:
        if sp:
            canv.setPageSize(self.pagesize)
        elif cp != dp:
            canv.setPageSize(doc.pagesize)
    for box in ('crop', 'art', 'trim', 'bleed'):
        size = getattr(self, box + 'Box', None)
        if size:
            canv.setCropBox(size, name=box)