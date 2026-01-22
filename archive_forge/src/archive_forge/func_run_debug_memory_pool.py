import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def run_debug_memory_pool(pool_factory, env_value):
    """
    Run a piece of code making an invalid memory write with the
    ARROW_DEBUG_MEMORY_POOL environment variable set to a specific value.
    """
    code = f'if 1:\n        import ctypes\n        import pyarrow as pa\n        # ARROW-16873: some Python installs enable faulthandler by default,\n        # which could dump a spurious stack trace if the following crashes\n        import faulthandler\n        faulthandler.disable()\n\n        pool = pa.{pool_factory}()\n        buf = pa.allocate_buffer(64, memory_pool=pool)\n\n        # Write memory out of bounds\n        ptr = ctypes.cast(buf.address, ctypes.POINTER(ctypes.c_ubyte))\n        ptr[64] = 0\n\n        del buf\n        '
    env = dict(os.environ)
    env['ARROW_DEBUG_MEMORY_POOL'] = env_value
    res = subprocess.run([sys.executable, '-c', code], env=env, universal_newlines=True, stderr=subprocess.PIPE)
    print(res.stderr, file=sys.stderr)
    return res