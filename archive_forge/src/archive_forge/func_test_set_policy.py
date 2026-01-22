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
def test_set_policy(get_module):
    get_handler_name = np.core.multiarray.get_handler_name
    get_handler_version = np.core.multiarray.get_handler_version
    orig_policy_name = get_handler_name()
    a = np.arange(10).reshape((2, 5))
    assert get_handler_name(a) is None
    assert get_handler_version(a) is None
    assert get_handler_name(a.base) == orig_policy_name
    assert get_handler_version(a.base) == 1
    orig_policy = get_module.set_secret_data_policy()
    b = np.arange(10).reshape((2, 5))
    assert get_handler_name(b) is None
    assert get_handler_version(b) is None
    assert get_handler_name(b.base) == 'secret_data_allocator'
    assert get_handler_version(b.base) == 1
    if orig_policy_name == 'default_allocator':
        get_module.set_old_policy(None)
        assert get_handler_name() == 'default_allocator'
    else:
        get_module.set_old_policy(orig_policy)
        assert get_handler_name() == orig_policy_name