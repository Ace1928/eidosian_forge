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
def handle_breakBefore(self, flowables):
    """preprocessing step to allow pageBreakBefore and frameBreakBefore attributes"""
    first = flowables[0]
    if hasattr(first, '_skipMeNextTime'):
        delattr(first, '_skipMeNextTime')
        return
    if hasattr(first, 'pageBreakBefore') and first.pageBreakBefore == 1:
        first._skipMeNextTime = 1
        first.insert(0, PageBreak())
        return
    if hasattr(first, 'style') and hasattr(first.style, 'pageBreakBefore') and (first.style.pageBreakBefore == 1):
        first._skipMeNextTime = 1
        flowables.insert(0, PageBreak())
        return
    if hasattr(first, 'frameBreakBefore') and first.frameBreakBefore == 1:
        first._skipMeNextTime = 1
        flowables.insert(0, FrameBreak())
        return
    if hasattr(first, 'style') and hasattr(first.style, 'frameBreakBefore') and (first.style.frameBreakBefore == 1):
        first._skipMeNextTime = 1
        flowables.insert(0, FrameBreak())
        return