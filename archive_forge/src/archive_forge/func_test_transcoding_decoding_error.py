import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
@pytest.mark.parametrize('src_encoding, dest_encoding', [('utf-8', 'utf-16'), ('utf-16', 'utf-8')])
def test_transcoding_decoding_error(src_encoding, dest_encoding):
    stream = pa.transcoding_input_stream(pa.BufferReader(b'\xff\xff\xff\xff'), src_encoding, dest_encoding)
    with pytest.raises(UnicodeError):
        stream.read(1)