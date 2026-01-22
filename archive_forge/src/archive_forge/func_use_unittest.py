import multiprocessing
import unittest
import warnings
import pytest
from monty.dev import deprecated, get_ncpus, install_excepthook, requires
@requires(unittest is not None, 'scipy is not present.')
def use_unittest():
    return 'success'