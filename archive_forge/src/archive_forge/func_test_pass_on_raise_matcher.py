import sys
from testtools import (
from testtools.matchers import (
def test_pass_on_raise_matcher(self):
    with ExpectedException(ValueError, AfterPreprocessing(str, Equals('test'))):
        raise ValueError('test')