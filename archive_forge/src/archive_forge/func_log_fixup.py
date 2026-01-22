import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
@contextmanager
def log_fixup():
    _hnsw._set_logger(sys.stdout)
    try:
        yield
    finally:
        _hnsw._reset_logger()