import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_models_get_different_ids(model_with_no_args):
    model1 = Linear()
    model2 = Linear()
    assert model1.id != model2.id