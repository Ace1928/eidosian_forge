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
def contour_vector_from_stats(stats):
    size = sqrt(abs(stats.area))
    return (copysign(size, stats.area), stats.meanX, stats.meanY, stats.stddevX * 2, stats.stddevY * 2, stats.correlation * size)