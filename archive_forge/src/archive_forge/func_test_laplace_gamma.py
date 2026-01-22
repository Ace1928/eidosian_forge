import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_laplace_gamma(laplace_bigram_model):
    assert laplace_bigram_model.gamma == 1