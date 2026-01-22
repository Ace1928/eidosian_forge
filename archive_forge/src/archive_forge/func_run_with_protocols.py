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
def run_with_protocols(self, meth, *args, **kwargs):
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        meth(proto, *args, **kwargs)