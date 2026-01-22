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
def initialize_read(self):
    """ Run when beginning read of variables

        Sets up readers from parameters in `self`
        """
    self._file_reader = VarReader5(self)
    self._matrix_reader = VarReader5(self)