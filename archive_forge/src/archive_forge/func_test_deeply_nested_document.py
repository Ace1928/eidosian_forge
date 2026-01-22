import os
import pytest
from bs4 import (
@pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5000587759190016', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5375146639360000', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5492400320282624'])
def test_deeply_nested_document(self, filename):
    self.fuzz_test_with_css(filename)