from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.basePen import AbstractPen, BasePen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, SegmentToPointPen
from fontTools.pens.recordingPen import RecordingPen, DecomposingRecordingPen
from fontTools.misc.transform import Transform
from collections import defaultdict, deque
from math import sqrt, copysign, atan2, pi
from enum import Enum
import itertools
import logging
def transform_from_stats(stats, inverse=False):
    a = stats.varianceX
    b = stats.covariance
    c = stats.varianceY
    delta = (((a - c) * 0.5) ** 2 + b * b) ** 0.5
    lambda1 = (a + c) * 0.5 + delta
    lambda2 = (a + c) * 0.5 - delta
    theta = atan2(lambda1 - a, b) if b != 0 else pi * 0.5 if a < c else 0
    trans = Transform()
    if lambda2 < 0:
        lambda2 = 0
    if inverse:
        trans = trans.translate(-stats.meanX, -stats.meanY)
        trans = trans.rotate(-theta)
        trans = trans.scale(1 / sqrt(lambda1), 1 / sqrt(lambda2))
    else:
        trans = trans.scale(sqrt(lambda1), sqrt(lambda2))
        trans = trans.rotate(theta)
        trans = trans.translate(stats.meanX, stats.meanY)
    return trans