from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_constructor_no_arguments(self):
    mismatch = Mismatch()
    self.assertThat(mismatch.describe, Raises(MatchesException(NotImplementedError)))
    self.assertEqual({}, mismatch.get_details())