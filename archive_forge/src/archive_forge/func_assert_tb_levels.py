from __future__ import annotations
import traceback
from contextlib import contextmanager
import pytest
import dask
from dask.utils import shorten_traceback
@contextmanager
def assert_tb_levels(expect):
    with pytest.raises(ZeroDivisionError) as e:
        yield
    frames = list(traceback.walk_tb(e.tb))
    frame_names = [frame[0].f_code.co_name for frame in frames]
    assert frame_names[0] == 'assert_tb_levels', frame_names
    assert frame_names[1:] == expect, frame_names