import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_existing_submodule(self):
    result = try_imports(['os.path'], object())
    import os
    self.assertThat(result, Is(os.path))