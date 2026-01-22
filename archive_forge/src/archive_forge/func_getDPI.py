import io
import os
import re
import struct
from xml.etree import ElementTree
def getDPI(filepath):
    """
    Return (x DPI, y DPI) for a given img file content
    no requirements
    :type filepath: Union[bytes, str, pathlib.Path]
    :rtype Tuple[int, int]
    """
    xDPI = -1
    yDPI = -1
    if not isinstance(filepath, bytes):
        filepath = str(filepath)
    with open(filepath, 'rb') as fhandle:
        head = fhandle.read(24)
        size = len(head)
        if size >= 10 and head[:6] in (b'GIF87a', b'GIF89a'):
            pass
        elif size >= 24 and head.startswith(b'\x89PNG\r\n\x1a\n'):
            chunkOffset = 8
            chunk = head[8:]
            while True:
                chunkType = chunk[4:8]
                if chunkType == b'pHYs':
                    try:
                        xDensity, yDensity, unit = struct.unpack('>LLB', chunk[8:])
                    except struct.error:
                        raise ValueError('Invalid PNG file')
                    if unit:
                        xDPI = _convertToDPI(xDensity, _UNIT_1M)
                        yDPI = _convertToDPI(yDensity, _UNIT_1M)
                    else:
                        xDPI = xDensity
                        yDPI = yDensity
                    break
                elif chunkType == b'IDAT':
                    break
                else:
                    try:
                        dataSize, = struct.unpack('>L', chunk[0:4])
                    except struct.error:
                        raise ValueError('Invalid PNG file')
                    chunkOffset += dataSize + 12
                    fhandle.seek(chunkOffset)
                    chunk = fhandle.read(17)
        elif size >= 2 and head.startswith(b'\xff\xd8'):
            try:
                fhandle.seek(0)
                size = 2
                ftype = 0
                while not 192 <= ftype <= 207:
                    if ftype == 224:
                        fhandle.seek(7, 1)
                        unit, xDensity, yDensity = struct.unpack('>BHH', fhandle.read(5))
                        if unit == 1 or unit == 0:
                            xDPI = xDensity
                            yDPI = yDensity
                        elif unit == 2:
                            xDPI = _convertToDPI(xDensity, _UNIT_CM)
                            yDPI = _convertToDPI(yDensity, _UNIT_CM)
                        break
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 255:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
            except struct.error:
                raise ValueError('Invalid JPEG file')
        elif size >= 12 and head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
            fhandle.seek(32)
            headerSize = struct.unpack('>L', fhandle.read(4))[0] - 8
            fhandle.seek(4, 1)
            foundResBox = False
            try:
                while headerSize > 0:
                    boxHeader = fhandle.read(8)
                    boxType = boxHeader[4:]
                    if boxType == b'res ':
                        foundResBox = True
                        headerSize -= 8
                        break
                    boxSize, = struct.unpack('>L', boxHeader[:4])
                    fhandle.seek(boxSize - 8, 1)
                    headerSize -= boxSize
                if foundResBox:
                    while headerSize > 0:
                        boxHeader = fhandle.read(8)
                        boxType = boxHeader[4:]
                        if boxType == b'resd':
                            yDensity, xDensity, yUnit, xUnit = struct.unpack('>HHBB', fhandle.read(10))
                            xDPI = _convertToDPI(xDensity, xUnit)
                            yDPI = _convertToDPI(yDensity, yUnit)
                            break
                        boxSize, = struct.unpack('>L', boxHeader[:4])
                        fhandle.seek(boxSize - 8, 1)
                        headerSize -= boxSize
            except struct.error as e:
                raise ValueError('Invalid JPEG2000 file')
    return (xDPI, yDPI)