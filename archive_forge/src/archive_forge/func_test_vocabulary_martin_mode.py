import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_vocabulary_martin_mode(self):
    """Tests all words from the test vocabulary provided by M Porter

        The sample vocabulary and output were sourced from
        https://tartarus.org/martin/PorterStemmer/voc.txt and
        https://tartarus.org/martin/PorterStemmer/output.txt
        and are linked to from the Porter Stemmer algorithm's homepage
        at https://tartarus.org/martin/PorterStemmer/
        """
    with closing(data.find('stemmers/porter_test/porter_martin_output.txt').open(encoding='utf-8')) as fp:
        self._test_against_expected_output(PorterStemmer.MARTIN_EXTENSIONS, fp.read().splitlines())