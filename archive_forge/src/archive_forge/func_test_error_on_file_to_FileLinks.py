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
def test_error_on_file_to_FileLinks():
    """FileLinks: Raises error when passed file
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    pytest.raises(ValueError, display.FileLinks, tf1.name)