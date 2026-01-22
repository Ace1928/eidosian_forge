import itertools
import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arrays_equal
from ..array_sequence import ArraySequence, concatenate, is_array_sequence
def test_creating_arraysequence_from_arraysequence(self):
    seq = ArraySequence(SEQ_DATA['data'])
    check_arr_seq(ArraySequence(seq), SEQ_DATA['data'])
    seq = ArraySequence()
    check_empty_arr_seq(ArraySequence(seq))