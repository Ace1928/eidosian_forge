import os
import pytest
from bs4 import (
@pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-4670634698080256', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5270998950477824'])
def test_soupsieve_errors(self, filename):
    self.fuzz_test_with_css(filename)