import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_oed_bug(self):
    """Test for bug https://github.com/nltk/nltk/issues/1581

        Ensures that 'oed' can be stemmed without throwing an error.
        """
    assert PorterStemmer().stem('oed') == 'o'