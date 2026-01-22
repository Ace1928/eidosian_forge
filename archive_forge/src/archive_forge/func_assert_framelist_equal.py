from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def assert_framelist_equal(list1, list2, *args, **kwargs):
    assert len(list1) == len(list2), f'lists are not of equal size len(list1) == {len(list1)}, len(list2) == {len(list2)}'
    msg = 'not all list elements are DataFrames'
    both_frames = all(map(lambda x, y: isinstance(x, DataFrame) and isinstance(y, DataFrame), list1, list2))
    assert both_frames, msg
    for frame_i, frame_j in zip(list1, list2):
        tm.assert_frame_equal(frame_i, frame_j, *args, **kwargs)
        assert not frame_i.empty, 'frames are both empty'