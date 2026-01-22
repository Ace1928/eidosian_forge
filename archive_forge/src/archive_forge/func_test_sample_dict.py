import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_sample_dict(self):
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    expected_dict = [(1, 2), (2, 3)]
    expected_dict_random = [(k, v) for k, v in d.items()]
    sampled_dict = utils.sample_dict(d, 2, False)
    self.assertEqual(sampled_dict, expected_dict)
    sampled_dict_random = utils.sample_dict(d, 2)
    if sampled_dict_random in expected_dict_random:
        self.assertTrue(True)