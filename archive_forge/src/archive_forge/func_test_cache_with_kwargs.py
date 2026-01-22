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
def test_cache_with_kwargs():
    global OFFSET
    OFFSET.clear()
    fn = cache(function_with_args)
    assert fn(a=0, b=0) == 0
    assert fn(a=0, b=0) == 0