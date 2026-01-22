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
def test_existing_path_FileLinks_repr_alt_formatter():
    """FileLinks: Calling repr() functions as expected w/ alt formatter
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)

    def fake_formatter(dirname, fnames, included_suffixes):
        return ['hello', 'world']
    fl = display.FileLinks(td, terminal_display_formatter=fake_formatter)
    actual = repr(fl)
    actual = actual.split('\n')
    actual.sort()
    expected = ['hello', 'world']
    expected.sort()
    assert actual == expected