import os
import zlib
import time  # noqa
import logging
import numpy as np
def make_style_change_record(self, lineStyle=None, fillStyle=None, moveTo=None):
    bits = BitArray()
    bits += '0'
    bits += '0'
    if lineStyle:
        bits += '1'
    else:
        bits += '0'
    if fillStyle:
        bits += '1'
    else:
        bits += '0'
    bits += '0'
    if moveTo:
        bits += '1'
    else:
        bits += '0'
    if moveTo:
        bits += twits2bits([moveTo[0], moveTo[1]])
    if fillStyle:
        bits += int2bits(fillStyle, 4)
    if lineStyle:
        bits += int2bits(lineStyle, 4)
    return bits