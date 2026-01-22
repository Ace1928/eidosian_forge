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
def test_recursive_FileLinks():
    """FileLinks: Does not recurse when recursive=False
    """
    td = mkdtemp()
    tf = NamedTemporaryFile(dir=td)
    subtd = mkdtemp(dir=td)
    subtf = NamedTemporaryFile(dir=subtd)
    fl = display.FileLinks(td)
    actual = str(fl)
    actual = actual.split('\n')
    assert len(actual) == 4, actual
    fl = display.FileLinks(td, recursive=False)
    actual = str(fl)
    actual = actual.split('\n')
    assert len(actual) == 2, actual