from __future__ import annotations
from typing import Dict, List, Union
import fontTools.otlLib.builder
from fontTools.designspaceLib import (
from fontTools.designspaceLib.types import Region, getVFUserRegion, locationInRegion
from fontTools.ttLib import TTFont
def getStatAxes(doc: DesignSpaceDocument, userRegion: Region) -> List[Dict]:
    """Return a list of axis dicts suitable for use as the ``axes``
    argument to :func:`fontTools.otlLib.builder.buildStatTable()`.

    .. versionadded:: 5.0
    """
    maxOrdering = max((axis.axisOrdering for axis in doc.axes if axis.axisOrdering is not None), default=-1)
    axisOrderings = []
    for axis in doc.axes:
        if axis.axisOrdering is not None:
            axisOrderings.append(axis.axisOrdering)
        else:
            maxOrdering += 1
            axisOrderings.append(maxOrdering)
    return [dict(tag=axis.tag, name={'en': axis.name, **axis.labelNames}, ordering=ordering, values=[_axisLabelToStatLocation(label) for label in axis.axisLabels if locationInRegion({axis.name: label.userValue}, userRegion)]) for axis, ordering in zip(doc.axes, axisOrderings)]