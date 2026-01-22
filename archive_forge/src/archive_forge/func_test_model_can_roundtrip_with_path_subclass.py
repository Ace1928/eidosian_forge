import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_model_can_roundtrip_with_path_subclass(model_with_no_args, pathy_fixture):
    path = pathy_fixture / 'thinc_model'
    model_with_no_args.to_disk(path)
    m2 = model_with_no_args.from_disk(path)
    assert model_with_no_args.to_bytes() == m2.to_bytes()