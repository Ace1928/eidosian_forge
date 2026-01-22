import itertools
import logging; log = logging.getLogger(__name__)
from passlib.tests.utils import TestCase
from passlib.pwd import genword, default_charsets
from passlib.pwd import genphrase
def test_wordset(self):
    """'wordset' & 'words' options"""
    results = genphrase(words=simple_words, returns=5000)
    self.assertResultContents(results, 5000, simple_words)
    results = genphrase(length=3, words=simple_words, returns=5000)
    self.assertResultContents(results, 5000, simple_words, unique=3 ** 3)
    self.assertRaises(TypeError, genphrase, words=simple_words, wordset='bip39')