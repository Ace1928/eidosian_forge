import os
import re
from copy import deepcopy
import numpy as np
from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder
def parse_AFNI_header(fobj):
    """
    Parses `fobj` to extract information from HEAD file

    Parameters
    ----------
    fobj : file-like object
        AFNI HEAD file object or filename. If file object, should
        implement at least ``read``

    Returns
    -------
    info : dict
        Dictionary containing AFNI-style key:value pairs from HEAD file

    Examples
    --------
    >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
    >>> info = parse_AFNI_header(fname)
    >>> print(info['BYTEORDER_STRING'])
    LSB_FIRST
    >>> print(info['BRICK_TYPES'])
    [1, 1, 1]
    """
    if isinstance(fobj, str):
        with open(fobj) as src:
            return parse_AFNI_header(src)
    head = fobj.read().split('\n\n')
    return {key: value for key, value in map(_unpack_var, head)}