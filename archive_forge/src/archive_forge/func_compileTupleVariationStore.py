from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def compileTupleVariationStore(variations, pointCount, axisTags, sharedTupleIndices, useSharedPoints=True):
    del pointCount
    newVariations = []
    pointDatas = []
    sharedPoints = None
    pointSetCount = defaultdict(int)
    for v in variations:
        points = v.getUsedPoints()
        if points is None:
            continue
        pointSetCount[points] += 1
        newVariations.append(v)
        pointDatas.append(points)
    variations = newVariations
    del newVariations
    if not variations:
        return (0, b'', b'')
    n = len(variations[0].coordinates)
    assert all((len(v.coordinates) == n for v in variations)), 'Variation sets have different sizes'
    compiledPoints = {pointSet: TupleVariation.compilePoints(pointSet) for pointSet in pointSetCount}
    tupleVariationCount = len(variations)
    tuples = []
    data = []
    if useSharedPoints:

        def key(pn):
            pointSet = pn[0]
            count = pn[1]
            return len(compiledPoints[pointSet]) * (count - 1)
        sharedPoints = max(pointSetCount.items(), key=key)[0]
        data.append(compiledPoints[sharedPoints])
        tupleVariationCount |= TUPLES_SHARE_POINT_NUMBERS
    pointDatas = [compiledPoints[points] if points != sharedPoints else b'' for points in pointDatas]
    for v, p in zip(variations, pointDatas):
        thisTuple, thisData = v.compile(axisTags, sharedTupleIndices, pointData=p)
        tuples.append(thisTuple)
        data.append(thisData)
    tuples = b''.join(tuples)
    data = b''.join(data)
    return (tupleVariationCount, tuples, data)