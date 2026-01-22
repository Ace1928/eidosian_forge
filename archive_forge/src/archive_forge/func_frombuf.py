from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
@classmethod
def frombuf(cls, buf, encoding, errors):
    """Construct a TarInfo object from a 512 byte bytes object.
        """
    if len(buf) == 0:
        raise EmptyHeaderError('empty header')
    if len(buf) != BLOCKSIZE:
        raise TruncatedHeaderError('truncated header')
    if buf.count(NUL) == BLOCKSIZE:
        raise EOFHeaderError('end of file header')
    chksum = nti(buf[148:156])
    if chksum not in calc_chksums(buf):
        raise InvalidHeaderError('bad checksum')
    obj = cls()
    obj.name = nts(buf[0:100], encoding, errors)
    obj.mode = nti(buf[100:108])
    obj.uid = nti(buf[108:116])
    obj.gid = nti(buf[116:124])
    obj.size = nti(buf[124:136])
    obj.mtime = nti(buf[136:148])
    obj.chksum = chksum
    obj.type = buf[156:157]
    obj.linkname = nts(buf[157:257], encoding, errors)
    obj.uname = nts(buf[265:297], encoding, errors)
    obj.gname = nts(buf[297:329], encoding, errors)
    obj.devmajor = nti(buf[329:337])
    obj.devminor = nti(buf[337:345])
    prefix = nts(buf[345:500], encoding, errors)
    if obj.type == AREGTYPE and obj.name.endswith('/'):
        obj.type = DIRTYPE
    if obj.type == GNUTYPE_SPARSE:
        pos = 386
        structs = []
        for i in range(4):
            try:
                offset = nti(buf[pos:pos + 12])
                numbytes = nti(buf[pos + 12:pos + 24])
            except ValueError:
                break
            structs.append((offset, numbytes))
            pos += 24
        isextended = bool(buf[482])
        origsize = nti(buf[483:495])
        obj._sparse_structs = (structs, isextended, origsize)
    if obj.isdir():
        obj.name = obj.name.rstrip('/')
    if prefix and obj.type not in GNU_TYPES:
        obj.name = prefix + '/' + obj.name
    return obj