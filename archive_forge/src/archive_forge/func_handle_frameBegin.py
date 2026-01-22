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
def handle_frameBegin(self, resume=0, pageTopFlowables=None):
    """What to do at the beginning of a frame"""
    f = self.frame
    if f._atTop:
        boundary = self.frame.showBoundary or self.showBoundary
        if boundary:
            self.frame.drawBoundary(self.canv, boundary)
    f._leftExtraIndent = self._leftExtraIndent
    f._rightExtraIndent = self._rightExtraIndent
    f._frameBGs = self._frameBGs
    if pageTopFlowables:
        self._hanging.extend(pageTopFlowables)
    if self._topFlowables:
        self._hanging.extend(self._topFlowables)