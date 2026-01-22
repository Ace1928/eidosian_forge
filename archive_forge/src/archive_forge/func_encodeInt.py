from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def encodeInt(value, fourByteOp=fourByteOp, bytechr=bytechr, pack=struct.pack, unpack=struct.unpack, twoByteOp=twoByteOp):
    if -107 <= value <= 107:
        code = bytechr(value + 139)
    elif 108 <= value <= 1131:
        value = value - 108
        code = bytechr((value >> 8) + 247) + bytechr(value & 255)
    elif -1131 <= value <= -108:
        value = -value - 108
        code = bytechr((value >> 8) + 251) + bytechr(value & 255)
    elif twoByteOp is not None and -32768 <= value <= 32767:
        code = twoByteOp + pack('>h', value)
    elif fourByteOp is None:
        log.warning('4-byte T2 number got passed to the IntType handler. This should happen only when reading in old XML files.\n')
        code = bytechr(255) + pack('>l', value)
    else:
        code = fourByteOp + pack('>l', value)
    return code