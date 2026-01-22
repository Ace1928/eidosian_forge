import numpy as np
from ...metaarray import MetaArray
def removePeriodic(data, f0=60.0, dt=None, harmonics=10, samples=4):
    if hasattr(data, 'implements') and data.implements('MetaArray'):
        data1 = data.asarray()
        if dt is None:
            times = data.xvals('Time')
            dt = times[1] - times[0]
    else:
        data1 = data
        if dt is None:
            raise Exception('Must specify dt for this data')
    ft = np.fft.fft(data1)
    df = 1.0 / (len(data1) * dt)
    for i in range(1, harmonics + 2):
        f = f0 * i
        ind1 = int(np.floor(f / df))
        ind2 = int(np.ceil(f / df)) + (samples - 1)
        if ind1 > len(ft) / 2.0:
            break
        mag = (abs(ft[ind1 - 1]) + abs(ft[ind2 + 1])) * 0.5
        for j in range(ind1, ind2 + 1):
            phase = np.angle(ft[j])
            re = mag * np.cos(phase)
            im = mag * np.sin(phase)
            ft[j] = re + im * 1j
            ft[len(ft) - j] = re - im * 1j
    data2 = np.fft.ifft(ft).real
    if hasattr(data, 'implements') and data.implements('MetaArray'):
        return metaarray.MetaArray(data2, info=data.infoCopy())
    else:
        return data2