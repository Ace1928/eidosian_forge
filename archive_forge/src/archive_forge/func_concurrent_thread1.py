import asyncio
import gc
import os
import pytest
import numpy as np
import threading
import warnings
from numpy.testing import extbuild, assert_warns, IS_WASM
import sys
def concurrent_thread1(get_module, event):
    get_module.set_secret_data_policy()
    assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'
    event.set()