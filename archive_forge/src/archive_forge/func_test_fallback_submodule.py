import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_fallback_submodule(self):
    result = try_imports(['os.doesntexist', 'os.path'])
    import os
    self.assertThat(result, Is(os.path))