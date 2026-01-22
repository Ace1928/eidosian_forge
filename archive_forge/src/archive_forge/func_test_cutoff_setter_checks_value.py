import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_cutoff_setter_checks_value(self):
    with self.assertRaises(ValueError) as exc_info:
        Vocabulary('abc', unk_cutoff=0)
    expected_error_msg = 'Cutoff value cannot be less than 1. Got: 0'
    self.assertEqual(expected_error_msg, str(exc_info.exception))