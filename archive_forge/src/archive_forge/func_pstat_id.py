import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def pstat_id(fs):
    return (fs.module, fs.lineno, fs.name)