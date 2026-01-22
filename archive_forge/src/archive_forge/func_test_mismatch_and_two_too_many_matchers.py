import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_mismatch_and_two_too_many_matchers(self):
    self.assertMismatchWithDescriptionMatching([3, 4], MatchesSetwise(Equals(0), Equals(1), Equals(2), Equals(3)), MatchesRegex('.*There was 1 mismatch and 2 extra matchers: Equals\\([012]\\), Equals\\([012]\\)', re.S))