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
def test_audio_data_without_normalization_raises_for_invalid_data(self):
    self.assertRaises(ValueError, lambda: display.Audio([1.001], rate=44100, normalize=False))
    self.assertRaises(ValueError, lambda: display.Audio([-1.001], rate=44100, normalize=False))