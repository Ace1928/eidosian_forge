import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
def silence_generator(frequency, sample_rate):
    while True:
        yield 0