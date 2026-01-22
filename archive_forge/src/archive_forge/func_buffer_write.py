import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def buffer_write(self, data, dtype):
    """Write audio data from a buffer/bytes object to the file.

        Writes the contents of *data* to the file at the current
        read/write position.
        This also advances the read/write position by the number of
        frames that were written and enlarges the file if necessary.

        Parameters
        ----------
        data : buffer or bytes
            A buffer or bytes object containing the audio data to be
            written.
        dtype : {'float64', 'float32', 'int32', 'int16'}
            The data type of the audio data stored in *data*.

        See Also
        --------
        .write, buffer_read

        """
    ctype = self._check_dtype(dtype)
    cdata, frames = self._check_buffer(data, ctype)
    written = self._cdata_io('write', cdata, ctype, frames)
    assert written == frames
    self._update_frames(written)