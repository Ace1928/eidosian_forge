import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def check_env_var(name, expected, *, expect_warning=False):
    code = f'if 1:\n        import pyarrow as pa\n\n        pool = pa.default_memory_pool()\n        assert pool.backend_name in {expected!r}, pool.backend_name\n        '
    env = dict(os.environ)
    env['ARROW_DEFAULT_MEMORY_POOL'] = name
    res = subprocess.run([sys.executable, '-c', code], env=env, universal_newlines=True, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        res.check_returncode()
    errlines = res.stderr.splitlines()
    if expect_warning:
        assert len(errlines) in (1, 2)
        if len(errlines) == 1:
            assert f"Unsupported backend '{name}'" in errlines[0]
        else:
            assert 'InitGoogleLogging()' in errlines[0]
            assert f"Unsupported backend '{name}'" in errlines[1]
    else:
        assert len(errlines) == 0