import sys
from testtools import (
from testtools.matchers import (
def test_raise_on_text_mismatch(self):
    try:
        with ExpectedException(ValueError, 'tes.'):
            raise ValueError('mismatch')
    except AssertionError:
        e = sys.exc_info()[1]
        self.assertEqual("'mismatch' does not match /tes./", str(e))
    else:
        self.fail('AssertionError not raised.')