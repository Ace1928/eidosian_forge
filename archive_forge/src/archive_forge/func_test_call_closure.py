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
def test_call_closure(self):
    from .serialize_usecases import closure
    inner = closure(1)
    self.run_with_protocols(self.check_call, inner, 6, (2, 3))