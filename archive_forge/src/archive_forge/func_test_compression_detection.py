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
@pytest.mark.parametrize(('path', 'expected_compression'), [('file.bz2', 'bz2'), ('file.lz4', 'lz4'), (pathlib.Path('file.gz'), 'gzip'), (pathlib.Path('path/to/file.zst'), 'zstd')])
def test_compression_detection(path, expected_compression):
    if not Codec.is_available(expected_compression):
        with pytest.raises(pa.lib.ArrowNotImplementedError):
            Codec.detect(path)
    else:
        codec = Codec.detect(path)
        assert isinstance(codec, Codec)
        assert codec.name == expected_compression