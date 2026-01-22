from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def checkAxisValuesExist(stat, axisValues, axisCoords):
    seen = set()
    designAxes = stat.DesignAxisRecord.Axis
    hasValues = set()
    for value in stat.AxisValueArray.AxisValue:
        if value.Format in (1, 2, 3):
            hasValues.add(designAxes[value.AxisIndex].AxisTag)
        elif value.Format == 4:
            for rec in value.AxisValueRecord:
                hasValues.add(designAxes[rec.AxisIndex].AxisTag)
    for axisValueTable in axisValues:
        axisValueFormat = axisValueTable.Format
        if axisValueTable.Format in (1, 2, 3):
            axisTag = designAxes[axisValueTable.AxisIndex].AxisTag
            if axisValueFormat == 2:
                axisValue = axisValueTable.NominalValue
            else:
                axisValue = axisValueTable.Value
            if axisTag in axisCoords and axisValue == axisCoords[axisTag]:
                seen.add(axisTag)
        elif axisValueTable.Format == 4:
            for rec in axisValueTable.AxisValueRecord:
                axisTag = designAxes[rec.AxisIndex].AxisTag
                if axisTag in axisCoords and rec.Value == axisCoords[axisTag]:
                    seen.add(axisTag)
    missingAxes = set(axisCoords) - seen & hasValues
    if missingAxes:
        missing = ', '.join((f"'{i}': {axisCoords[i]}" for i in missingAxes))
        raise ValueError(f'Cannot find Axis Values {{{missing}}}')