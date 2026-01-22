import multiprocessing
import unittest
import warnings
import pytest
from monty.dev import deprecated, get_ncpus, install_excepthook, requires
@requires(fictitious_mod is not None, 'fictitious_mod is not present.')
def use_fictitious_mod():
    print('success')