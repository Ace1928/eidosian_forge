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
@pytest.mark.skip(reason='Spawned process seems to share initialization state with parent.')
def test_call_error_when_ended_r():
    q = multiprocessing.Queue()
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=_call_with_ended_r, args=(q,))
    p.start()
    res = q.get()
    p.join()
    assert res[0]