import sys
import pytest
from . import util
@pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MinGW64 Gfortran (Issue #9673)')
def test_quoted_character(self):
    assert self.module.foo() == (b"'", b'"', b';', b'!', b'(', b')')