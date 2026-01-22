import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
def generate_seed():
    assert _nest_count > 0, 'random is not set up'
    return numpy.random.randint(2147483647)