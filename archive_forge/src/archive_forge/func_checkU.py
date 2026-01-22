import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def checkU(encryptionkey, U):
    decoded = computeU(encryptionkey, U)
    if decoded != PadString:
        if len(decoded) != len(PadString):
            raise ValueError("lengths don't match! (password failed)")
        raise ValueError("decode of U doesn't match fixed padstring (password failed)")