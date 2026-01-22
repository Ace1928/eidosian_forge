import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_russian(self):
    stemmer_russian = SnowballStemmer('russian')
    assert stemmer_russian.stem('авантненькая') == 'авантненьк'