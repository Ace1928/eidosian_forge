from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
def get_filters(plist):
    """ Extract a dictionary of active filters from a DCPL, along with
    their settings.

    Undocumented and subject to change without warning.
    """
    filters = {h5z.FILTER_DEFLATE: 'gzip', h5z.FILTER_SZIP: 'szip', h5z.FILTER_SHUFFLE: 'shuffle', h5z.FILTER_FLETCHER32: 'fletcher32', h5z.FILTER_LZF: 'lzf', h5z.FILTER_SCALEOFFSET: 'scaleoffset'}
    pipeline = {}
    nfilters = plist.get_nfilters()
    for i in range(nfilters):
        code, _, vals, _ = plist.get_filter(i)
        if code == h5z.FILTER_DEFLATE:
            vals = vals[0]
        elif code == h5z.FILTER_SZIP:
            mask, pixels = vals[0:2]
            if mask & h5z.SZIP_EC_OPTION_MASK:
                mask = 'ec'
            elif mask & h5z.SZIP_NN_OPTION_MASK:
                mask = 'nn'
            else:
                raise TypeError('Unknown SZIP configuration')
            vals = (mask, pixels)
        elif code == h5z.FILTER_LZF:
            vals = None
        elif len(vals) == 0:
            vals = None
        pipeline[filters.get(code, str(code))] = vals
    return pipeline