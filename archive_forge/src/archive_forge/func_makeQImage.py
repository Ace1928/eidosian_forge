from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def makeQImage(imgData, alpha=None, copy=True, transpose=True):
    """
    Turn an ARGB array into QImage.
    By default, the data is copied; changes to the array will not
    be reflected in the image. The image will be given a 'data' attribute
    pointing to the array which shares its data to prevent python
    freeing that memory while the image is in use.
    
    ============== ===================================================================
    **Arguments:**
    imgData        Array of data to convert. Must have shape (height, width),
                   (height, width, 3), or (height, width, 4). If transpose is
                   True, then the first two axes are swapped. The array dtype
                   must be ubyte. For 2D arrays, the value is interpreted as 
                   greyscale. For 3D arrays, the order of values in the 3rd
                   axis must be (b, g, r, a). 
    alpha          If the input array is 3D and *alpha* is True, the QImage 
                   returned will have format ARGB32. If False,
                   the format will be RGB32. By default, _alpha_ is True if
                   array.shape[2] == 4.
    copy           If True, the data is copied before converting to QImage.
                   If False, the new QImage points directly to the data in the array.
                   Note that the array must be contiguous for this to work
                   (see numpy.ascontiguousarray).
    transpose      If True (the default), the array x/y axes are transposed before 
                   creating the image. Note that Qt expects the axes to be in 
                   (height, width) order whereas pyqtgraph usually prefers the 
                   opposite.
    ============== ===================================================================    
    """
    profile = debug.Profiler()
    copied = False
    if imgData.ndim == 2:
        imgFormat = QtGui.QImage.Format.Format_Grayscale8
    elif imgData.ndim == 3:
        if alpha is None:
            alpha = imgData.shape[2] == 4
        if imgData.shape[2] == 3:
            if copy is True:
                d2 = np.empty(imgData.shape[:2] + (4,), dtype=imgData.dtype)
                d2[:, :, :3] = imgData
                d2[:, :, 3] = 255
                imgData = d2
                copied = True
            else:
                raise Exception('Array has only 3 channels; cannot make QImage without copying.')
        profile('add alpha channel')
        if alpha:
            imgFormat = QtGui.QImage.Format.Format_ARGB32
        else:
            imgFormat = QtGui.QImage.Format.Format_RGB32
    else:
        raise TypeError('Image array must have ndim = 2 or 3.')
    if transpose:
        imgData = imgData.transpose((1, 0, 2))
    if not imgData.flags['C_CONTIGUOUS']:
        if copy is False:
            extra = ' (try setting transpose=False)' if transpose else ''
            raise Exception('Array is not contiguous; cannot make QImage without copying.' + extra)
        imgData = np.ascontiguousarray(imgData)
        copied = True
    profile('ascontiguousarray')
    if copy is True and copied is False:
        imgData = imgData.copy()
    profile('copy')
    return ndarray_to_qimage(imgData, imgFormat)