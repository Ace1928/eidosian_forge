import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@cuda_ipc
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_IPC(size):
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    arr, cbuf = make_random_buffer(size=size, target='device')
    ipc_handle = cbuf.export_for_ipc()
    handle_buffer = ipc_handle.serialize()
    p = ctx.Process(target=other_process_for_test_IPC, args=(handle_buffer, arr))
    p.start()
    p.join()
    assert p.exitcode == 0