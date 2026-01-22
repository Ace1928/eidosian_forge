import sys
import os
import struct
import logging
import numpy as np
def splitSerieIfRequired(serie, series, progressIndicator):
    """
    Split the serie in multiple series if this is required. The choice
    is based on examing the image position relative to the previous
    image. If it differs too much, it is assumed that there is a new
    dataset. This can happen for example in unspitted gated CT data.
    """
    serie._sort()
    L = serie._entries
    ds1 = L[0]
    if 'ImagePositionPatient' not in ds1:
        return
    L2 = [[ds1]]
    distance = 0
    for index in range(1, len(L)):
        ds2 = L[index]
        pos1 = float(ds1.ImagePositionPatient[2])
        pos2 = float(ds2.ImagePositionPatient[2])
        newDist = abs(pos1 - pos2)
        if distance and newDist > 2.1 * distance:
            L2.append([])
            distance = 0
        else:
            if distance and newDist > 1.5 * distance:
                progressIndicator.write('Warning: missing file after %r' % ds1._filename)
            distance = newDist
        L2[-1].append(ds2)
        ds1 = ds2
    if len(L2) > 1:
        i = series.index(serie)
        series2insert = []
        for L in L2:
            newSerie = DicomSeries(serie.suid, progressIndicator)
            newSerie._entries = L
            series2insert.append(newSerie)
        for newSerie in reversed(series2insert):
            series.insert(i, newSerie)
        series.remove(serie)