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
def write_empty_struct(self):
    self.write_header((1, 1), mxSTRUCT_CLASS)
    self.write_element(np.array(1, dtype=np.int32))
    self.write_element(np.array([], dtype=np.int8))