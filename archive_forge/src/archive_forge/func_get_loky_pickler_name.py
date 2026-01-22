import copyreg
import io
import functools
import types
import sys
import os
from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL
def get_loky_pickler_name():
    global _loky_pickler_name
    return _loky_pickler_name