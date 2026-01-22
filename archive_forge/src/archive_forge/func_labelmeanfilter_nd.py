from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def labelmeanfilter_nd(y, x):
    labelsunique = np.arange(np.max(y) + 1)
    labmeansdata = []
    labmeans = []
    for xx in x.T:
        labelmeans = np.array(ndimage.mean(xx, labels=y, index=labelsunique))
        labmeansdata.append(labelmeans[y])
        labmeans.append(labelmeans)
    labelcount = np.array(ndimage.histogram(y, labelsunique[0], labelsunique[-1] + 1, 1, labels=y, index=labelsunique))
    return (labelcount, np.array(labmeans), np.array(labmeansdata).T)