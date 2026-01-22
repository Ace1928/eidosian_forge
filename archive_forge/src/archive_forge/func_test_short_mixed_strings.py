import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_short_mixed_strings(self):
    b, u = (_b('§'), '§')
    mismatch = _BinaryMismatch(b, '!~', u)
    self.assertEqual(mismatch.describe(), f'{b!r} !~ {u!r}')