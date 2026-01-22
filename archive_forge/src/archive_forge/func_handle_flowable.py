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
def handle_flowable(self, flowables):
    """try to handle one flowable from the front of list flowables."""
    self.filterFlowables(flowables)
    f = flowables[0]
    if f:
        self.handle_breakBefore(flowables)
        self.handle_keepWithNext(flowables)
        f = flowables[0]
    del flowables[0]
    if f is None:
        return
    if isinstance(f, PageBreak):
        npt = f.nextTemplate
        if npt and (not self._samePT(npt)):
            npt = NextPageTemplate(npt)
            npt.apply(self)
            self.afterFlowable(npt)
        if isinstance(f, SlowPageBreak):
            self.handle_pageBreak(slow=1)
        else:
            self.handle_pageBreak()
        self.afterFlowable(f)
    elif isinstance(f, ActionFlowable):
        f.apply(self)
        self.afterFlowable(f)
    else:
        frame = self.frame
        canv = self.canv
        if frame.add(f, canv, trySplit=self.allowSplitting):
            if not isinstance(f, FrameActionFlowable):
                self._curPageFlowableCount += 1
                self.afterFlowable(f)
            _addGeneratedContent(flowables, frame)
        else:
            if self.allowSplitting:
                S = frame.split(f, canv)
                n = len(S)
            else:
                n = 0
            if n:
                if not isinstance(S[0], (PageBreak, SlowPageBreak, ActionFlowable, DDIndenter)):
                    if not frame.add(S[0], canv, trySplit=0):
                        ident = 'Splitting error(n==%d) on page %d in\n%s\nS[0]=%s' % (n, self.page, self._fIdent(f, 60, frame), self._fIdent(S[0], 60, frame))
                        raise LayoutError(ident)
                    self._curPageFlowableCount += 1
                    self.afterFlowable(S[0])
                    flowables[0:0] = S[1:]
                    _addGeneratedContent(flowables, frame)
                else:
                    flowables[0:0] = S
            else:
                if hasattr(f, '_postponed'):
                    ident = 'Flowable %s%s too large on page %d in frame %r%s of template %r' % (self._fIdent(f, 60, frame), _fSizeString(f), self.page, self.frame.id, self.frame._aSpaceString(), self.pageTemplate.id)
                    raise LayoutError(ident)
                f._postponed = 1
                mbe = getattr(self, '_multiBuildEdits', None)
                if mbe:
                    mbe((delattr, f, '_postponed'))
                flowables.insert(0, f)
                self.handle_frameEnd()