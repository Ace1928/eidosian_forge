from __future__ import absolute_import, division, print_function
import errno
import os
import subprocess
import sys
import tempfile
import numpy as np
from ..utils import string_types
from ..audio.signal import Signal
def decode_to_disk(infile, fmt='f32le', sample_rate=None, num_channels=1, skip=None, max_len=None, outfile=None, tmp_dir=None, tmp_suffix=None, cmd='ffmpeg'):
    """
    Decode the given audio file to another file.

    Parameters
    ----------
    infile : str
        Name of the audio sound file to decode.
    fmt : {'f32le', 's16le'}, optional
        Format of the samples:
        - 'f32le' for float32, little-endian,
        - 's16le' for signed 16-bit int, little-endian.
    sample_rate : int, optional
        Sample rate to re-sample the signal to (if set) [Hz].
    num_channels : int, optional
        Number of channels to reduce the signal to.
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    outfile : str, optional
        The file to decode the sound file to; if not given, a temporary file
        will be created.
    tmp_dir : str, optional
        The directory to create the temporary file in (if no `outfile` is
        given).
    tmp_suffix : str, optional
        The file suffix for the temporary file if no `outfile` is given; e.g.
        ".pcm" (including the dot).
    cmd : {'ffmpeg', 'avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).

    Returns
    -------
    outfile : str
        The output file name.

    """
    if not isinstance(infile, string_types):
        raise ValueError('only file names are supported as `infile`, not %s.' % infile)
    if outfile is None:
        f = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=tmp_suffix)
        f.close()
        outfile = f.name
        delete_on_fail = True
    else:
        delete_on_fail = False
    if not isinstance(outfile, string_types):
        raise ValueError('only file names are supported as `outfile`, not %s.' % outfile)
    try:
        call = _ffmpeg_call(infile, outfile, fmt, sample_rate, num_channels, skip, max_len, cmd)
        subprocess.check_call(call)
    except Exception:
        if delete_on_fail:
            os.unlink(outfile)
        raise
    return outfile