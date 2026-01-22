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
def loadImageFromJPEG(self, imageFile):
    try:
        try:
            info = pdfutils.readJPEGInfo(imageFile)
        finally:
            imageFile.seek(0)
    except:
        return False
    self.width, self.height = (info[0], info[1])
    self.bitsPerComponent = 8
    if info[2] == 1:
        self.colorSpace = 'DeviceGray'
    elif info[2] == 3:
        self.colorSpace = 'DeviceRGB'
    else:
        self.colorSpace = 'DeviceCMYK'
        self._dotrans = 1
    self.streamContent = imageFile.read()
    if rl_config.useA85:
        self.streamContent = asciiBase85Encode(self.streamContent)
        self._filters = ('ASCII85Decode', 'DCTDecode')
    else:
        self._filters = ('DCTDecode',)
    self.mask = None
    return True