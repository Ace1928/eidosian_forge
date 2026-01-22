import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def readJPEGInfo(image):
    """Read width, height and number of components from open JPEG file."""
    import struct
    from reportlab.pdfbase.pdfdoc import PDFError
    validMarkers = [192, 193, 194]
    noParamMarkers = [208, 209, 210, 211, 212, 213, 214, 215, 216, 1]
    unsupportedMarkers = [195, 197, 198, 199, 200, 201, 202, 203, 205, 206, 207]
    dpi = (72, 72)
    done = 0
    while not done:
        x = struct.unpack('B', image.read(1))
        if x[0] == 255:
            x = struct.unpack('B', image.read(1))
            if x[0] in validMarkers:
                image.seek(2, 1)
                x = struct.unpack('B', image.read(1))
                if x[0] != 8:
                    raise PDFError('JPEG must have 8 bits per component')
                y = struct.unpack('BB', image.read(2))
                height = (y[0] << 8) + y[1]
                y = struct.unpack('BB', image.read(2))
                width = (y[0] << 8) + y[1]
                y = struct.unpack('B', image.read(1))
                color = y[0]
                return (width, height, color, dpi)
            elif x[0] == 224:
                x = struct.unpack('BB', image.read(2))
                n = (x[0] << 8) + x[1] - 2
                x = image.read(n)
                y = struct.unpack('BB', x[10:12])
                x = struct.unpack('BB', x[8:10])
                dpi = ((x[0] << 8) + x[1], (y[0] << 8) + y[1])
            elif x[0] in unsupportedMarkers:
                raise PDFError('JPEG Unsupported JPEG marker: %0.2x' % x[0])
            elif x[0] not in noParamMarkers:
                x = struct.unpack('BB', image.read(2))
                image.seek((x[0] << 8) + x[1] - 2, 1)