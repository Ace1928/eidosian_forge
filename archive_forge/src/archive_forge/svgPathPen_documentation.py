from typing import Callable
from fontTools.pens.basePen import BasePen

        >>> pen = SVGPathPen(None)
        >>> pen.endPath()
        >>> pen._commands
        []
        