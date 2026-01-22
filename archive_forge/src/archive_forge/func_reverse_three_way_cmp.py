import functools
import sys
import types
import warnings
import unittest
@staticmethod
def reverse_three_way_cmp(a, b):
    return unittest.util.three_way_cmp(b, a)