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
def put_variables(self, mdict, write_header=None):
    """ Write variables in `mdict` to stream

        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` returns name, contents pairs where
           ``name`` which will appear in the matlab workspace in file load, and
           ``contents`` is something writeable to a matlab file, such as a NumPy
           array.
        write_header : {None, True, False}, optional
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        """
    if write_header is None:
        write_header = self.file_stream.tell() == 0
    if write_header:
        self.write_file_header()
    self._matrix_writer = VarWriter5(self)
    for name, var in mdict.items():
        if name[0] == '_':
            continue
        is_global = name in self.global_vars
        if self.do_compression:
            stream = BytesIO()
            self._matrix_writer.file_stream = stream
            self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
            out_str = zlib.compress(stream.getvalue())
            tag = np.empty((), NDT_TAG_FULL)
            tag['mdtype'] = miCOMPRESSED
            tag['byte_count'] = len(out_str)
            self.file_stream.write(tag.tobytes())
            self.file_stream.write(out_str)
        else:
            self._matrix_writer.write_top(var, name.encode('latin1'), is_global)