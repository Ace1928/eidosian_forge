import cmath
import contextlib
from collections import defaultdict
import enum
import gc
import math
import platform
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import io
import ctypes
import multiprocessing as mp
import warnings
import traceback
from contextlib import contextmanager
import uuid
import importlib
import types as pytypes
from functools import cached_property
import numpy as np
from numba import testing, types
from numba.core import errors, typing, utils, config, cpu
from numba.core.typing import cffi_utils
from numba.core.compiler import (compile_extra, Flags,
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
import unittest
from numba.core.runtime import rtsys
from numba.np import numpy_support
from numba.core.runtime import _nrt_python as _nrt
from numba.core.extending import (
from numba.core.datamodel.models import OpaqueModel
def print_azure_matrix():
    """This is a utility function that prints out the map of NumPy to Python
    versions and how many of that combination are being tested across all the
    declared config for azure-pipelines. It is useful to run when updating the
    azure-pipelines config to be able to quickly see what the coverage is."""
    import yaml
    from yaml import Loader
    base_path = os.path.dirname(os.path.abspath(__file__))
    azure_pipe = os.path.join(base_path, '..', '..', 'azure-pipelines.yml')
    if not os.path.isfile(azure_pipe):
        raise RuntimeError("'azure-pipelines.yml' is not available")
    with open(os.path.abspath(azure_pipe), 'rt') as f:
        data = f.read()
    pipe_yml = yaml.load(data, Loader=Loader)
    templates = pipe_yml['jobs']
    py2np_map = defaultdict(lambda: defaultdict(int))
    for tmplt in templates[:2]:
        matrix = tmplt['parameters']['matrix']
        for setup in matrix.values():
            py2np_map[setup['NUMPY']][setup['PYTHON']] += 1
    winpath = ['..', '..', 'buildscripts', 'azure', 'azure-windows.yml']
    azure_windows = os.path.join(base_path, *winpath)
    if not os.path.isfile(azure_windows):
        raise RuntimeError("'azure-windows.yml' is not available")
    with open(os.path.abspath(azure_windows), 'rt') as f:
        data = f.read()
    windows_yml = yaml.load(data, Loader=Loader)
    matrix = windows_yml['jobs'][0]['strategy']['matrix']
    for setup in matrix.values():
        py2np_map[setup['NUMPY']][setup['PYTHON']] += 1
    print('NumPy | Python | Count')
    print('-----------------------')
    for npver, pys in sorted(py2np_map.items()):
        for pyver, count in pys.items():
            print(f' {npver} |  {pyver:<4}  |   {count}')
    rev_map = defaultdict(lambda: defaultdict(int))
    for npver, pys in sorted(py2np_map.items()):
        for pyver, count in pys.items():
            rev_map[pyver][npver] = count
    print('\nPython | NumPy | Count')
    print('-----------------------')
    sorter = lambda x: int(x[0].split('.')[1])
    for pyver, nps in sorted(rev_map.items(), key=sorter):
        for npver, count in nps.items():
            print(f' {pyver:<4} |  {npver}  |   {count}')