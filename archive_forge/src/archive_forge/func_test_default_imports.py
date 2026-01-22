from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
def test_default_imports():
    """
    Startup time: `import dask` should not import too many modules.
    """
    code = 'if 1:\n        import dask\n        import sys\n\n        print(sorted(sys.modules))\n        '
    out = subprocess.check_output([sys.executable, '-c', code])
    modules = set(eval(out.decode()))
    assert 'dask' in modules
    blacklist = ['dask.array', 'dask.dataframe', 'numpy', 'pandas', 'partd', 's3fs', 'distributed']
    for mod in blacklist:
        assert mod not in modules