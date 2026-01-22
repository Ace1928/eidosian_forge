import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_short_objects(self):
    o1, o2 = (self.CustomRepr('a'), self.CustomRepr('b'))
    mismatch = _BinaryMismatch(o1, '!~', o2)
    self.assertEqual(mismatch.describe(), f'{o1!r} !~ {o2!r}')