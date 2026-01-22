import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def encryptionkey(password, OwnerKey, Permissions, FileId1, revision=None):
    revision = checkRevision(revision)
    password = asBytes(password) + PadString
    password = password[:32]
    p = Permissions
    permissionsString = b''
    for i in range(4):
        byte = p & 255
        p = p >> 8
        permissionsString += int2Byte(byte % 256)
    hash = md5(asBytes(password))
    hash.update(asBytes(OwnerKey))
    hash.update(asBytes(permissionsString))
    hash.update(asBytes(FileId1))
    md5output = hash.digest()
    if revision == 2:
        key = md5output[:5]
    elif revision == 3:
        for x in range(50):
            md5output = md5(md5output).digest()
        key = md5output[:16]
    if DEBUG:
        print('encryptionkey(%s,%s,%s,%s,%s)==>%s' % tuple([hexText(str(x)) for x in (password, OwnerKey, Permissions, FileId1, revision, key)]))
    return key