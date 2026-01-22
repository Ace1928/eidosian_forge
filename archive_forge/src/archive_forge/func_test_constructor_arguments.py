from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_constructor_arguments(self):
    mismatch = Mismatch('some description', {'detail': 'things'})
    self.assertEqual('some description', mismatch.describe())
    self.assertEqual({'detail': 'things'}, mismatch.get_details())