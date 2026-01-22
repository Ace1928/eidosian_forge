from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
def qCurveTo(self, *points):
    n = len(points) - 1
    assert n >= 0
    if points[-1] is None:
        x, y = points[-2]
        nx, ny = points[0]
        impliedStartPoint = (0.5 * (x + nx), 0.5 * (y + ny))
        self.__currentPoint = impliedStartPoint
        self._moveTo(impliedStartPoint)
        points = points[:-1] + (impliedStartPoint,)
    if n > 0:
        _qCurveToOne = self._qCurveToOne
        for pt1, pt2 in decomposeQuadraticSegment(points):
            _qCurveToOne(pt1, pt2)
            self.__currentPoint = pt2
    else:
        self.lineTo(points[0])