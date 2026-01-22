import datetime as dt
import io
import pathlib
import time
from collections import Counter
import numpy as np
import pandas as pd
import param
import pytest
import requests
from panel.io.cache import _find_hash_func, cache
from panel.io.state import set_curdoc, state
from panel.tests.util import serve_and_wait
@pytest.mark.xdist_group('cache')
@diskcache_available
def test_disk_cache():
    global OFFSET
    OFFSET.clear()
    fn = cache(function_with_args, to_disk=True)
    assert fn(0, 0) == 0
    assert pathlib.Path('./cache').exists()
    assert list(pathlib.Path('./cache').glob('*'))
    assert fn(0, 0) == 0
    fn.clear()
    assert fn(0, 0) == 1