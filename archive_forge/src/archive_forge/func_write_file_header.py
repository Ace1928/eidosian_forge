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
def write_file_header(self):
    hdr = np.zeros((), NDT_FILE_HDR)
    hdr['description'] = f'MATLAB 5.0 MAT-file Platform: {os.name}, Created on: {time.asctime()}'
    hdr['version'] = 256
    hdr['endian_test'] = np.ndarray(shape=(), dtype='S2', buffer=np.uint16(19785))
    self.file_stream.write(hdr.tobytes())