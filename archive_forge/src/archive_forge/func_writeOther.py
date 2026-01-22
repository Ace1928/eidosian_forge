import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def writeOther(path, data, dohex=False):
    chunks = findEncryptedChunks(data)
    with open(path, 'wb') as f:
        hexlinelen = HEXLINELENGTH // 2
        for isEncrypted, chunk in chunks:
            if isEncrypted:
                code = 2
            else:
                code = 1
            if code == 2 and dohex:
                while chunk:
                    f.write(eexec.hexString(chunk[:hexlinelen]))
                    f.write(b'\r')
                    chunk = chunk[hexlinelen:]
            else:
                f.write(chunk)