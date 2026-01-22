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
@pytest.mark.parametrize('envir', (None, rinterface.globalenv, rinterface.ListSexpVector([])))
def test_evalr(envir):
    res = rinterface.evalr('1 + 2', envir=envir)
    assert tuple(res) == (3,)