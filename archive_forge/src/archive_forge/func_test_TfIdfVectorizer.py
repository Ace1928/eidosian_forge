import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_TfIdfVectorizer(self) -> None:
    self._test_op_upgrade('TfIdfVectorizer', 9, [[3]], [[5]], attrs={'max_gram_length': 3, 'max_skip_count': 1, 'min_gram_length': 2, 'mode': 'TFIDF', 'ngram_counts': [0, 20], 'ngram_indexes': [3, 4]})