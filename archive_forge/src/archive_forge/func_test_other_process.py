import contextlib
import gc
import pickle
import runpy
import subprocess
import sys
import unittest
from multiprocessing import get_context
import numba
from numba.core.errors import TypingError
from numba.tests.support import TestCase
from numba.core.target_extension import resolve_dispatcher_from_str
from numba.cloudpickle import dumps, loads
def test_other_process(self):
    """
        Check that reconstructing doesn't depend on resources already
        instantiated in the original process.
        """
    from .serialize_usecases import closure_calling_other_closure
    func = closure_calling_other_closure(3.0)
    pickled = pickle.dumps(func)
    code = 'if 1:\n            import pickle\n\n            data = {pickled!r}\n            func = pickle.loads(data)\n            res = func(4.0)\n            assert res == 8.0, res\n            '.format(**locals())
    subprocess.check_call([sys.executable, '-c', code])