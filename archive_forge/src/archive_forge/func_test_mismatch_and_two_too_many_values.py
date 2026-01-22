import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_mismatch_and_two_too_many_values(self):
    self.assertMismatchWithDescriptionMatching([2, 3, 4, 5], MatchesSetwise(Equals(1), Equals(2)), MatchesRegex('.*There was 1 mismatch and 2 extra values: \\[[145], [145]\\]', re.S))