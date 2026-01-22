import array
import socket
import struct
from os_ken.lib import addrconv
def fletcher_checksum(data, offset):
    """
    Fletcher Checksum -- Refer to RFC1008

    calling with offset == _FLETCHER_CHECKSUM_VALIDATE will validate the
    checksum without modifying the buffer; a valid checksum returns 0.
    """
    c0 = 0
    c1 = 0
    pos = 0
    length = len(data)
    data = bytearray(data)
    data[offset:offset + 2] = [0] * 2
    while pos < length:
        tlen = min(length - pos, _MODX)
        for d in data[pos:pos + tlen]:
            c0 += d
            c1 += c0
        c0 %= 255
        c1 %= 255
        pos += tlen
    x = ((length - offset - 1) * c0 - c1) % 255
    if x <= 0:
        x += 255
    y = 510 - c0 - x
    if y > 255:
        y -= 255
    data[offset] = x
    data[offset + 1] = y
    return x << 8 | y & 255