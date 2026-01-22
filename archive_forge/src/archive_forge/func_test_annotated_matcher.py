import sys
from testtools import (
from testtools.matchers import (
def test_annotated_matcher(self):

    def die():
        with ExpectedException(ValueError, 'bar', msg='foo'):
            pass
    exc = self.assertRaises(AssertionError, die)
    self.assertThat(exc.args[0], EndsWith(': foo'))