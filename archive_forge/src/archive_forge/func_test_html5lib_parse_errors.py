import os
import pytest
from bs4 import (
@pytest.mark.skip(reason='html5lib-specific problems')
@pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-6306874195312640'])
def test_html5lib_parse_errors(self, filename):
    self.fuzz_test_with_css(filename)