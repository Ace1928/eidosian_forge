from tempfile import NamedTemporaryFile, mkdtemp
from os.path import split, join as pjoin, dirname
import pathlib
from unittest import TestCase, mock
import struct
import wave
from io import BytesIO
import pytest
from IPython.lib import display
from IPython.testing.decorators import skipif_not_numpy
def read_wav(data):
    with wave.open(BytesIO(data)) as wave_file:
        wave_data = wave_file.readframes(wave_file.getnframes())
        num_samples = wave_file.getnframes() * wave_file.getnchannels()
        return struct.unpack('<%sh' % num_samples, wave_data)