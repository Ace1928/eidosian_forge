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
def test_rternalize_map_ellipsis_args_kwargs_error():

    def f(x, *args, y=2, **kwargs):
        pass
    with pytest.raises(ValueError):
        rfun = rinterface.rternalize(f, signature=True)