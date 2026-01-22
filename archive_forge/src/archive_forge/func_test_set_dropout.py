import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_set_dropout():
    model = Dropout()
    assert model.attrs['dropout_rate'] == 0.0
    set_dropout_rate(model, 0.2)
    assert model.attrs['dropout_rate'] == 0.2