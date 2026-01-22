import asyncio
import gc
import os
import pytest
import numpy as np
import threading
import warnings
from numpy.testing import extbuild, assert_warns, IS_WASM
import sys
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_thread_locality(get_module):
    orig_policy_name = np.core.multiarray.get_handler_name()
    event = threading.Event()
    concurrent_task1 = threading.Thread(target=concurrent_thread1, args=(get_module, event))
    concurrent_task2 = threading.Thread(target=concurrent_thread2, args=(get_module, event))
    concurrent_task1.start()
    concurrent_task2.start()
    concurrent_task1.join()
    concurrent_task2.join()
    assert np.core.multiarray.get_handler_name() == orig_policy_name