import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_vocabulary_original_mode(self):
    with closing(data.find('stemmers/porter_test/porter_original_output.txt').open(encoding='utf-8')) as fp:
        self._test_against_expected_output(PorterStemmer.ORIGINAL_ALGORITHM, fp.read().splitlines())
    self._test_against_expected_output(PorterStemmer.ORIGINAL_ALGORITHM, data.find('stemmers/porter_test/porter_original_output.txt').open(encoding='utf-8').read().splitlines())