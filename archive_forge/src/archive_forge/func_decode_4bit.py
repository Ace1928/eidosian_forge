import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
def decode_4bit(bits, palette, width, height, pitch, pitch_sign):
    rgb_pitch = ((pitch << 1) + 1 & ~1) * 3
    buffer = (ctypes.c_ubyte * (height * rgb_pitch))()
    i = 0
    for row in bits:
        for packed in row:
            for index in ((packed & 240) >> 4, packed & 15):
                rgb = palette[index]
                buffer[i] = rgb.rgbRed
                buffer[i + 1] = rgb.rgbGreen
                buffer[i + 2] = rgb.rgbBlue
                i += 3
    return ImageData(width, height, 'RGB', buffer, pitch_sign * rgb_pitch)