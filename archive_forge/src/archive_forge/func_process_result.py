import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def process_result(result):
    return result * 2