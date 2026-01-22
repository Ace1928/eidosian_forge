import numpy as np
from ...metaarray import MetaArray
def butterworthFilter(data, wPass, wStop=None, gPass=2.0, gStop=20.0, order=1, dt=None, btype='low', bidir=True):
    """return data passed through bessel filter"""
    try:
        import scipy.signal
    except ImportError:
        raise Exception('butterworthFilter() requires the package scipy.signal.')
    if dt is None:
        try:
            tvals = data.xvals('Time')
            dt = (tvals[-1] - tvals[0]) / (len(tvals) - 1)
        except:
            dt = 1.0
    if wStop is None:
        wStop = wPass * 2.0
    ord, Wn = scipy.signal.buttord(wPass * dt * 2.0, wStop * dt * 2.0, gPass, gStop)
    b, a = scipy.signal.butter(ord, Wn, btype=btype)
    return applyFilter(data, b, a, bidir=bidir)