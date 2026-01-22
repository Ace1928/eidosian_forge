import sys
from testtools import (
from testtools.matchers import (
def test_pass_on_raise_any_message(self):
    with ExpectedException(ValueError):
        raise ValueError('whatever')