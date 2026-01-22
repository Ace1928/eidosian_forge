from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_if_message_no_message(self):
    matcher = Equals(1)
    not_annotated = Annotate.if_message('', matcher)
    self.assertIs(matcher, not_annotated)