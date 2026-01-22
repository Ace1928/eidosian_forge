import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_nonexistent_submodule(self):
    marker = object()
    result = try_imports(['os.doesntexist'], marker)
    self.assertThat(result, Is(marker))