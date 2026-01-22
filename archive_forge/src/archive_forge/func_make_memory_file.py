from gitdb import OStream
import sys
import random
from array import array
from io import BytesIO
import glob
import unittest
import tempfile
import shutil
import os
import gc
import logging
from functools import wraps
def make_memory_file(size_in_bytes, randomize=False):
    """:return: tuple(size_of_stream, stream)
    :param randomize: try to produce a very random stream"""
    d = make_bytes(size_in_bytes, randomize)
    return (len(d), BytesIO(d))