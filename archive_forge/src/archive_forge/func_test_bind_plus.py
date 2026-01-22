import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_bind_plus():
    with Model.define_operators({'+': lambda a, b: (a.name, b.name)}):
        m = create_model(name='a') + create_model(name='b')
        assert m == ('a', 'b')