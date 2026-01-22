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
def function_with_args(a, b):
    global OFFSET
    offset = OFFSET.get((a, b), 0)
    result = a + b + offset
    OFFSET[a, b] = offset + 1
    return result