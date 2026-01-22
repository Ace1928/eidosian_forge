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
@pytest.mark.parametrize('envir', (None, rinterface.globalenv))
@pytest.mark.parametrize('expr,visibility', (('x <- 1', False), ('1', True)))
def test_evalr_expr_with_visible(envir, expr, visibility):
    value, vis = rinterface.evalr_expr_with_visible(rinterface.parse(expr), envir=envir)
    assert vis[0] == visibility