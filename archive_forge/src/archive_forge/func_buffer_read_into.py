import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def buffer_read_into(self, buffer, dtype):
    """Read from the file into a given buffer object.

        Fills the given *buffer* with frames in the given data format
        starting at the current read/write position (which can be
        changed with `seek()`) until the buffer is full or the end
        of the file is reached.  This advances the read/write position
        by the number of frames that were read.

        Parameters
        ----------
        buffer : writable buffer
            Audio frames from the file are written to this buffer.
        dtype : {'float64', 'float32', 'int32', 'int16'}
            The data type of *buffer*.

        Returns
        -------
        int
            The number of frames that were read from the file.
            This can be less than the size of *buffer*.
            The rest of the buffer is not filled with meaningful data.

        See Also
        --------
        buffer_read, .read

        """
    ctype = self._check_dtype(dtype)
    cdata, frames = self._check_buffer(buffer, ctype)
    frames = self._cdata_io('read', cdata, ctype, frames)
    return frames