import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def computeO(userPassword, ownerPassword, revision):
    from reportlab.lib.arciv import ArcIV
    assert revision in (2, 3), 'Unknown algorithm revision %s' % revision
    if not ownerPassword:
        ownerPassword = userPassword
    ownerPad = asBytes(ownerPassword) + PadString
    ownerPad = ownerPad[0:32]
    password = asBytes(userPassword) + PadString
    userPad = password[:32]
    digest = md5(ownerPad).digest()
    if DEBUG:
        print('PadString=%s\nownerPad=%s\npassword=%s\nuserPad=%s\ndigest=%s\nrevision=%s' % (ascii(PadString), ascii(ownerPad), ascii(password), ascii(userPad), ascii(digest), revision))
    if revision == 2:
        O = ArcIV(digest[:5]).encode(userPad)
    elif revision == 3:
        for i in range(50):
            digest = md5(digest).digest()
        digest = digest[:16]
        O = userPad
        for i in range(20):
            thisKey = xorKey(i, digest)
            O = ArcIV(thisKey).encode(O)
    if DEBUG:
        print('computeO(%s,%s,%s)==>%s' % tuple([hexText(str(x)) for x in (userPassword, ownerPassword, revision, O)]))
    return O