from __future__ import absolute_import, division, print_function
import errno
import os
import subprocess
import sys
import tempfile
import numpy as np
from ..utils import string_types
from ..audio.signal import Signal
def decode_to_memory(infile, fmt='f32le', sample_rate=None, num_channels=1, skip=None, max_len=None, cmd='ffmpeg'):
    """
    Decode the given audio and return it as a binary string representation.

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
    cmd : {'ffmpeg', 'avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).

    Returns
    -------
    samples : str
        Binary string representation of the audio samples.

    """
    if not isinstance(infile, (string_types, Signal)):
        raise ValueError('only file names or Signal instances are supported as `infile`, not %s.' % infile)
    _, proc = decode_to_pipe(infile, fmt=fmt, sample_rate=sample_rate, num_channels=num_channels, skip=skip, max_len=max_len, cmd=cmd)
    if isinstance(infile, Signal):
        try:
            signal, _ = proc.communicate(np.getbuffer(infile))
        except AttributeError:
            mv = memoryview(infile)
            signal, _ = proc.communicate(mv.cast('b'))
    else:
        signal, _ = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return signal