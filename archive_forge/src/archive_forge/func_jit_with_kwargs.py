import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def jit_with_kwargs(name, kwarg_list):
    call_args_with_kwargs = ','.join([f'{kw}={kw}' for kw in kwarg_list])
    signature = ','.join(kwarg_list)
    code = f'def func({signature}):\n        return {name}({call_args_with_kwargs})\n'
    pyfunc = compile_function('func', code, globals())
    return jit(nopython=True)(pyfunc)