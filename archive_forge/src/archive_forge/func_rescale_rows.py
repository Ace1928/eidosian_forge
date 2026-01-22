import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def rescale_rows(rows, rescale):
    """
    Take each row in rows (an iterator) and yield
    a fresh row with the pixels scaled according to
    the rescale parameters in the list `rescale`.
    Each element of `rescale` is a tuple of
    (source_bitdepth, target_bitdepth),
    with one element per channel.
    """
    fs = [float(2 ** s[1] - 1) / float(2 ** s[0] - 1) for s in rescale]
    target_bitdepths = set((s[1] for s in rescale))
    assert len(target_bitdepths) == 1
    target_bitdepth, = target_bitdepths
    typecode = 'BH'[target_bitdepth > 8]
    n_chans = len(rescale)
    for row in rows:
        rescaled_row = array(typecode, iter(row))
        for i in range(n_chans):
            channel = array(typecode, (int(round(fs[i] * x)) for x in row[i::n_chans]))
            rescaled_row[i::n_chans] = channel
        yield rescaled_row