import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_doesnt_exist(self):
    marker = object()
    result = try_imports(['doesntexist'], marker)
    self.assertThat(result, Is(marker))