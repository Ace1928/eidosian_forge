import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def checkRevision(revision):
    if revision is None:
        strength = rl_config.encryptionStrength
        if strength == 40:
            revision = 2
        elif strength == 128:
            revision = 3
        elif strength == 256:
            if not pyaes:
                raise ValueError('strength==256 is not supported as package pyaes is not importable')
            revision = 5
        else:
            raise ValueError('Unknown encryption strength=%s' % repr(strength))
    return revision