import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_strided_windows2(self):
    input_arr = np.arange(10)
    out = utils.strided_windows(input_arr, 5)
    expected = self.arr10_5.copy()
    self._assert_arrays_equal(expected, out)
    out[0, 0] = 10
    self.assertEqual(10, input_arr[0], 'should make view rather than copy')