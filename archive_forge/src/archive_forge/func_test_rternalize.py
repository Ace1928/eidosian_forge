import gc
import multiprocessing
import os
import pickle
import pytest
from rpy2 import rinterface
import rpy2
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import signal
import sys
import subprocess
import tempfile
import textwrap
import time
@pytest.mark.parametrize('signature', ((True,), (False,)))
def test_rternalize(signature):

    def f(x, y):
        return x[0] + y[0]
    rfun = rinterface.rternalize(f, signature=signature)
    res = rfun(1, 2)
    assert res[0] == 3