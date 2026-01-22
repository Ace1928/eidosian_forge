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
def check_thread_safety(self, extract_randomness):
    """
        When initializing the PRNG the same way, each thread
        should produce the same sequence of random numbers,
        using independent states, regardless of parallel
        execution.
        """
    results = self.extract_in_threads(15, extract_randomness, seed=42)
    self.check_several_outputs(results, same_expected=True)