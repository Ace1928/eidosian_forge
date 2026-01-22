import sys
from scipy._lib._testutils import _parse_size, _get_mem_available
import pytest
def test__mem_available():
    available = _get_mem_available()
    if sys.platform.startswith('linux'):
        assert available >= 0
    else:
        assert available is None or available >= 0