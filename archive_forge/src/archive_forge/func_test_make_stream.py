import os
import zlib
from io import BytesIO
from tempfile import mkstemp
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._streams import (make_stream,
def test_make_stream():
    with setup_test_file() as (fs, gs, cs):
        assert_(isinstance(make_stream(gs), GenericStream))