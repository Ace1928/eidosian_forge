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
def fastLoad(self, directory):
    started = time.clock()
    f = open(os.path.join(directory, self.name + '.fastmap'), 'rb')
    self._mapFileHash = marshal.load(f)
    self._codeSpaceRanges = marshal.load(f)
    self._notDefRanges = marshal.load(f)
    self._cmap = marshal.load(f)
    f.close()
    finished = time.clock()