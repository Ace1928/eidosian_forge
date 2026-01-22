import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
def test_tempmemmap(monkeypatch):
    registration_counter = RegistrationCounter()
    monkeypatch.setattr(atexit, 'register', registration_counter)
    input_array = np.ones(3)
    with TempMemmap(input_array) as data:
        check_memmap(input_array, data)
        temp_folder = os.path.dirname(data.filename)
    if os.name != 'nt':
        assert not os.path.exists(temp_folder)
    assert registration_counter.nb_calls == 1
    mmap_mode = 'r+'
    with TempMemmap(input_array, mmap_mode=mmap_mode) as data:
        check_memmap(input_array, data, mmap_mode=mmap_mode)
        temp_folder = os.path.dirname(data.filename)
    if os.name != 'nt':
        assert not os.path.exists(temp_folder)
    assert registration_counter.nb_calls == 2