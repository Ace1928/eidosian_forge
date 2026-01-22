import re
import ctypes
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.image import AbstractImage, Texture
def decode_dxt1_rgba(data, width, height):
    out = (ctypes.c_ubyte * (width * height * 4))()
    pitch = width << 2
    image_offset = 0
    for c0_lo, c0_hi, c1_lo, c1_hi, b0, b1, b2, b3 in split_8byte.findall(data):
        color0 = ord(c0_lo) | ord(c0_hi) << 8
        color1 = ord(c1_lo) | ord(c1_hi) << 8
        bits = ord(b0) | ord(b1) << 8 | ord(b2) << 16 | ord(b3) << 24
        r0 = color0 & 31
        g0 = (color0 & 2016) >> 5
        b0 = (color0 & 63488) >> 11
        r1 = color1 & 31
        g1 = (color1 & 2016) >> 5
        b1 = (color1 & 63488) >> 11
        i = image_offset
        for y in range(4):
            for x in range(4):
                code = bits & 3
                a = 255
                if code == 0:
                    r, g, b = (r0, g0, b0)
                elif code == 1:
                    r, g, b = (r1, g1, b1)
                elif code == 3 and color0 <= color1:
                    r = g = b = a = 0
                elif code == 2 and color0 > color1:
                    r = (2 * r0 + r1) // 3
                    g = (2 * g0 + g1) // 3
                    b = (2 * b0 + b1) // 3
                elif code == 3 and color0 > color1:
                    r = (r0 + 2 * r1) // 3
                    g = (g0 + 2 * g1) // 3
                    b = (b0 + 2 * b1) // 3
                else:
                    assert code == 2 and color0 <= color1
                    r = (r0 + r1) // 2
                    g = (g0 + g1) // 2
                    b = (b0 + b1) // 2
                out[i] = b << 3
                out[i + 1] = g << 2
                out[i + 2] = r << 3
                out[i + 3] = a << 4
                bits >>= 2
                i += 4
            i += pitch - 16
        advance_row = (image_offset + 16) % pitch == 0
        image_offset += pitch * 3 * advance_row + 16
    return PackedImageData(width, height, GL_RGBA, GL_UNSIGNED_BYTE, out)