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
def cvtdict(self, d, escape=1):
    """transform dict args from python form to pdf string rep as needed"""
    Rect = d['Rect']
    Quad = d['QuadPoints']
    Color = d['C']
    if not isinstance(Rect, str):
        d['Rect'] = PDFArray(Rect).format(d, IND=b' ')
    if not isinstance(Quad, str):
        d['QuadPoints'] = PDFArray(Quad).format(d, IND=b' ')
    if not isinstance(Color, str):
        d['C'] = PDFArray(Color).format(d, IND=b' ')
    d['Contents'] = PDFString(d['Contents'], escape)
    return d