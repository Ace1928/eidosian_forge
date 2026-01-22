import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
def test_generate_oov_text_seed(mle_trigram_model):
    assert mle_trigram_model.generate(text_seed=('aliens',), random_seed=3) == mle_trigram_model.generate(text_seed=('<UNK>',), random_seed=3)