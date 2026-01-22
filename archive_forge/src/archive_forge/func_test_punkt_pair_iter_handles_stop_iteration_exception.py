from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_punkt_pair_iter_handles_stop_iteration_exception(self):
    it = iter([])
    gen = punkt._pair_iter(it)
    list(gen)