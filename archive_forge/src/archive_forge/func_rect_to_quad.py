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
def rect_to_quad(Rect):
    """
    Utility method to convert a Rect to a QuadPoint
    """
    return [Rect[0], Rect[1], Rect[2], Rect[1], Rect[0], Rect[3], Rect[2], Rect[3]]