import math
from operator import itemgetter
import pytest
from nltk.lm import (
from nltk.lm.preprocessing import padded_everygrams
@pytest.fixture
def lidstone_bigram_model(bigram_training_data, vocabulary):
    model = Lidstone(0.1, order=2, vocabulary=vocabulary)
    model.fit(bigram_training_data)
    return model