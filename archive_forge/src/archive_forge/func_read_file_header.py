import os
import time
import sys
import zlib
from io import BytesIO
import warnings
import numpy as np
import scipy.sparse
from ._byteordercodes import native_code, swapped_code
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio5_utils import VarReader5
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
from ._streams import ZlibInputStream
def read_file_header(self):
    """ Read in mat 5 file header """
    hdict = {}
    hdr_dtype = MDTYPES[self.byte_order]['dtypes']['file_header']
    hdr = read_dtype(self.mat_stream, hdr_dtype)
    hdict['__header__'] = hdr['description'].item().strip(b' \t\n\x00')
    v_major = hdr['version'] >> 8
    v_minor = hdr['version'] & 255
    hdict['__version__'] = '%d.%d' % (v_major, v_minor)
    return hdict