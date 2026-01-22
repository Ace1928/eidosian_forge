import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
def inPage(self):
    """specify the current object as a page (enables reference binding and other page features)"""
    if self.inObject is not None:
        if self.inObject == 'page':
            return
        raise ValueError("can't go in page already in object %s" % self.inObject)
    self.inObject = 'page'