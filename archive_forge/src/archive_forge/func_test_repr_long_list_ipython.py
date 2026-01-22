import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_repr_long_list_ipython(self):
    args = ['-m', 'IPython', '--quiet', '--quick', '--no-banner', '--colors=NoColor', '-c']
    base_cmd = [sys.executable] + args
    try:
        subprocess.check_output(base_cmd + ['--version'])
    except subprocess.CalledProcessError as e:
        self.skipTest('ipython not found: return code %d' % e.returncode)
    repr_cmd = [' '.join(['import sys;', 'from numba.typed import List;', 'res = repr(List(range(1005)));', 'sys.stderr.write(res);'])]
    cmd = base_cmd + repr_cmd
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    l = List(range(1005))
    expected = f'{typeof(l)}([{', '.join(map(str, l[:1000]))}, ...])'
    self.assertEqual(expected, err)