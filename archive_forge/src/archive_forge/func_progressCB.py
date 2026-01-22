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
def progressCB(typ, value):
    """Example prototype for progress monitoring.

    This aims to provide info about what is going on
    during a big job.  It should enable, for example, a reasonably
    smooth progress bar to be drawn.  We design the argument
    signature to be predictable and conducive to programming in
    other (type safe) languages.  If set, this will be called
    repeatedly with pairs of values.  The first is a string
    indicating the type of call; the second is a numeric value.

    typ 'STARTING', value = 0
    typ 'SIZE_EST', value = numeric estimate of job size
    typ 'PASS', value = number of this rendering pass
    typ 'PROGRESS', value = number between 0 and SIZE_EST
    typ 'PAGE', value = page number of page
    type 'FINISHED', value = 0

    The sequence is
        STARTING - always called once
        SIZE_EST - always called once
        PROGRESS - called often
        PAGE - called often when page is emitted
        FINISHED - called when really, really finished

    some juggling is needed to accurately estimate numbers of
    pages in pageDrawing mode.

    NOTE: the SIZE_EST is a guess.  It is possible that the
    PROGRESS value may slightly exceed it, or may even step
    back a little on rare occasions.  The only way to be
    really accurate would be to do two passes, and I don't
    want to take that performance hit.
    """
    print('PROGRESS MONITOR:  %-10s   %d' % (typ, value))