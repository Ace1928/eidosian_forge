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
@skipif_not_numpy
def test_audio_from_numpy_array(self):
    test_tone = get_test_tone()
    audio = display.Audio(test_tone, rate=44100)
    assert len(read_wav(audio.data)) == len(test_tone)