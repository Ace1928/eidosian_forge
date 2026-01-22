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
def translateNames(self, canvas, object):
    """recursively translate tree of names into tree of destinations"""
    destinationnamestotitles = self.destinationnamestotitles
    destinationstotitles = self.destinationstotitles
    closedict = self.closedict
    if isStr(object):
        if not isUnicode(object):
            object = object.decode('utf8')
        destination = canvas._bookmarkReference(object)
        title = object
        if object in destinationnamestotitles:
            title = destinationnamestotitles[object]
        else:
            destinationnamestotitles[title] = title
        destinationstotitles[destination] = title
        if object in closedict:
            closedict[destination] = 1
        return {object: canvas._bookmarkReference(object)}
    if isSeq(object):
        L = []
        for o in object:
            L.append(self.translateNames(canvas, o))
        if isinstance(object, tuple):
            return tuple(L)
        return L
    raise TypeError('in outline, destination name must be string: got a %s' % type(object))