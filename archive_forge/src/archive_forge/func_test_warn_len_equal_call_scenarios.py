import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_warn_len_equal_call_scenarios():

    class mod:
        pass
    mod_inst = mod()
    assert_warn_len_equal(mod=mod_inst, n_in_context=0)

    class mod:

        def __init__(self):
            self.__warningregistry__ = {'warning1': 1, 'warning2': 2}
    mod_inst = mod()
    assert_warn_len_equal(mod=mod_inst, n_in_context=2)