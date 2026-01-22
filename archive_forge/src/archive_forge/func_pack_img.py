from collections import namedtuple
from multiprocessing import current_process
import ctypes
import struct
import numbers
import numpy as np
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
def pack_img(header, img, quality=95, img_fmt='.jpg'):
    """Pack an image into ``MXImageRecord``.

    Parameters
    ----------
    header : IRHeader
        Header of the image record.
        ``header.label`` can be a number or an array. See more detail in ``IRHeader``.
    img : numpy.ndarray
        Image to be packed.
    quality : int
        Quality for JPEG encoding in range 1-100, or compression for PNG encoding in range 1-9.
    img_fmt : str
        Encoding of the image (.jpg for JPEG, .png for PNG).

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    >>> id = 2574
    >>> header = mx.recordio.IRHeader(0, label, id, 0)
    >>> img = cv2.imread('test.jpg')
    >>> packed_s = mx.recordio.pack_img(header, img)
    """
    assert cv2 is not None
    jpg_formats = ['.JPG', '.JPEG']
    png_formats = ['.PNG']
    encode_params = None
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, 'failed to encode image'
    return pack(header, buf.tostring())