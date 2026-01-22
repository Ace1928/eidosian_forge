import os
import marshal
import time
from hashlib import md5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase._cidfontdata import allowedTypeFaces, allowedEncodings, CIDFontInfo, \
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib.rl_accel import escapePDF
from reportlab.rl_config import CMapSearchPath
from reportlab.lib.utils import isSeq, isBytes
def findCMapFile(name):
    """Returns full filename, or raises error"""
    for dirname in CMapSearchPath:
        cmapfile = dirname + os.sep + name
        if os.path.isfile(cmapfile):
            return cmapfile
    raise IOError('CMAP file for encodings "%s" not found!' % name)