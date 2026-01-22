from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def segmentPointAtT(seg, t):
    if len(seg) == 2:
        return linePointAtT(*seg, t)
    elif len(seg) == 3:
        return quadraticPointAtT(*seg, t)
    elif len(seg) == 4:
        return cubicPointAtT(*seg, t)
    raise ValueError('Unknown curve degree')