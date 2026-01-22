import itertools
import logging; log = logging.getLogger(__name__)
from passlib.tests.utils import TestCase
from passlib.pwd import genword, default_charsets
from passlib.pwd import genphrase
def test_charset(self):
    """'charset' & 'chars' options"""
    results = genword(charset='hex', returns=5000)
    self.assertResultContents(results, 5000, hex)
    results = genword(length=3, chars='abc', returns=5000)
    self.assertResultContents(results, 5000, 'abc', unique=27)
    self.assertRaises(TypeError, genword, chars='abc', charset='hex')