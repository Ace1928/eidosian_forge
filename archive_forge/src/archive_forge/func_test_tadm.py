import pytest
from nltk import classify
def test_tadm():
    assert_classifier_correct('TADM')