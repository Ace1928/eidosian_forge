import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_lowercase_option(self):
    """Test for improvement on https://github.com/nltk/nltk/issues/2507

        Ensures that stems are lowercased when `to_lowercase=True`
        """
    porter = PorterStemmer()
    assert porter.stem('On') == 'on'
    assert porter.stem('I') == 'i'
    assert porter.stem('I', to_lowercase=False) == 'I'
    assert porter.stem('Github') == 'github'
    assert porter.stem('Github', to_lowercase=False) == 'Github'