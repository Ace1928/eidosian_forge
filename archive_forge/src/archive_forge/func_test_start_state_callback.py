from __future__ import annotations
from dask.callbacks import Callback
from dask.local import get_sync
from dask.threaded import get as get_threaded
from dask.utils_test import add
def test_start_state_callback():
    flag = [False]

    class MyCallback(Callback):

        def _start_state(self, dsk, state):
            flag[0] = True
            assert dsk['x'] == 1
            assert len(state['cache']) == 1
    with MyCallback():
        get_sync({'x': 1}, 'x')
    assert flag[0] is True